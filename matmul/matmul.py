from tensorflow.python.client import timeline
import json
import time
import numpy as np
import os
from hostlist import expand_hostlist
import tensorflow as tf

tf.app.flags.DEFINE_integer('size', 8192, 'Size of matrix.')
tf.app.flags.DEFINE_integer('tile_size', 4096, 'Size of matrix.')
tf.app.flags.DEFINE_integer('dequeue_batch_size', 16, 'Number of tiles a reducer collects per dequeue.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')
tf.app.flags.DEFINE_integer('num_reducers', 2, 'Number of reducers, at least two. The rest of the processes rest idle.')
tf.app.flags.DEFINE_integer('num_tests', 5, 'Number of tests to repeat.')
tf.app.flags.DEFINE_string('protocol', 'grpc+verbs', 'Transfer protocol.')
tf.app.flags.DEFINE_boolean('debug', False, 'Debug.')
FLAGS = tf.app.flags.FLAGS

# from https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def build_barrier(num):
    with tf.variable_scope('barrier'):
        with tf.device('/job:reducer/task:0/cpu:0'):
            counter = tf.get_variable(name='counter', shape=[], initializer=tf.zeros_initializer(), dtype=tf.int32, use_resource=True)
            increase_counter = tf.assign_add(counter, 1)
            c = lambda i: tf.not_equal(tf.mod(tf.add(i, num), num), 0)
            b = lambda i: counter
            with tf.control_dependencies([increase_counter]):
                barrier = tf.while_loop(c, b, [counter])
    return barrier

def build_dataset(task_index, num_workers):
    def generator():
        N = int(FLAGS.size)
        num_blocks = int(N/FLAGS.tile_size)

        for i in range(num_blocks):
            for j in range(num_blocks):
                for k in range(num_blocks):
                    yield [i, j], './data/'+str(N)+'/'+str(FLAGS.tile_size)+'/A_'+str(i)+'_'+str(k)+'.npy', './data/'+str(N)+'/'+str(FLAGS.tile_size)+'/B_'+str(k)+'_'+str(j)+'.npy'

    def map_func(target, a_path, b_path):
        tile_a = np.load(a_path.decode())
        tile_b = np.load(b_path.decode())
        return target.astype(np.float32), tile_a.astype(np.float32), tile_b.astype(np.float32)

    dataset = tf.data.Dataset.from_generator(generator,
                                             (tf.float32, tf.string, tf.string),
                                             (tf.TensorShape([2]), tf.TensorShape([]), tf.TensorShape([])))
    dataset = dataset.shard(num_workers, task_index)
    dataset = dataset.map(lambda target, a_path, b_path: tuple(tf.py_func(map_func, [target, a_path, b_path], [tf.float32, tf.float32, tf.float32,])))
    dataset = dataset.prefetch(2)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    iterator = dataset.make_initializable_iterator()

    return iterator

def create_queue(num_workers):
    num_blocks = int(FLAGS.size/FLAGS.tile_size)
    incoming = []
    outgoing = []
    with tf.device('/job:reducer/task:0'):
        queue = tf.FIFOQueue(capacity=FLAGS.dequeue_batch_size*FLAGS.dequeue_batch_size, dtypes=(tf.float32, tf.float32), shapes=[[2], [FLAGS.tile_size, FLAGS.tile_size]], name='even', shared_name='even')
    incoming.append(queue)
    outgoing.append(queue.dequeue_many(FLAGS.dequeue_batch_size))

    with tf.device('/job:reducer/task:0'):
        queue = tf.FIFOQueue(capacity=FLAGS.dequeue_batch_size*FLAGS.dequeue_batch_size, dtypes=(tf.float32, tf.float32), shapes=[[2], [FLAGS.tile_size, FLAGS.tile_size]], name='odd', shared_name='odd')
    incoming.append(queue)
    outgoing.append(queue.dequeue_many(FLAGS.dequeue_batch_size))

    return incoming, outgoing

def create_graph(task_index, num_workers):
    num_blocks = int(FLAGS.size/FLAGS.tile_size)

    incoming, outgoing = create_queue(num_workers)
    iterator = build_dataset(task_index, num_workers)
    idx, tile_a, tile_b = iterator.get_next()

    with tf.device('/job:worker/task:%d/gpu:0' % task_index):
        compute = tf.matmul(tile_a, tile_b)
        tile = tf.add(tf.multiply(idx[0], num_blocks), idx[1])
    with tf.device('/job:worker/task:%d/cpu:0' % task_index):
        queue_id = tf.mod(tile, 2)
        target = tf.equal(queue_id, 0)
        compute = tf.cond(tf.equal(tf.mod(tile, 2), 0), lambda: incoming[0].enqueue((idx, compute)), lambda: incoming[1].enqueue((idx, compute)))
#    with tf.control_dependencies([compute]):
#        compute = tf.no_op()

    return compute, iterator.initializer

def create_cluster(job_name, task_index):
    rank = int( os.environ['SLURM_PROCID'] )
    num_reducer = FLAGS.num_reducers
    num_gpus = FLAGS.num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank%num_gpus)

    tf_hostlist = []
    for host in expand_hostlist(os.environ['SLURM_NODELIST']):
        for gpu_id in range(num_gpus):
            tf_hostlist.append("%s:%d" % (host, 8888+gpu_id))

    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=FLAGS.debug)
    cluster = tf.train.ClusterSpec({ "reducer": tf_hostlist[0:FLAGS.num_reducers], "worker": tf_hostlist[FLAGS.num_reducers:] })

    return config, cluster

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    num_tasks = int(os.environ['SLURM_NTASKS'])
    num_workers = num_tasks - FLAGS.num_reducers
    rank = int( os.environ['SLURM_PROCID'] )
    if rank < FLAGS.num_reducers:
        job_name = 'reducer'
        task_index = rank
    else:
        job_name = 'worker'
        task_index = rank - FLAGS.num_reducers

    N = int(FLAGS.size)
    num_blocks = int(N/FLAGS.tile_size)
    num_tiles = int((N/FLAGS.tile_size)*(N/FLAGS.tile_size))
    print('matmul size: '+str(N)+'; tile size: '+str(FLAGS.tile_size))
#
#    cluster = tf.train.ClusterSpec({ 'reducer': tf_hostlist[0:num_reducers], 'worker': tf_hostlist[num_reducers:num_tasks] })
#    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)

    barrier = build_barrier(num_tasks)

    compute, init_iterator = create_graph(task_index, num_workers)

    print('job: '+str(job_name)+' idx: '+str(task_index))
    if job_name == 'worker':
        config, cluster = create_cluster('worker', num_tasks-FLAGS.num_reducers)
        server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, config=config, protocol=FLAGS.protocol)
    
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index==0),
                                               config=config) as sess:
            for repeat in range(FLAGS.num_tests):
                sess.run(init_iterator)

                timeline_saver = TimeLiner()
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                sess.run(barrier)
                for i in range(int(num_blocks*num_blocks*num_blocks/num_workers)):
                    if FLAGS.debug is True:
                        sess.run(compute, run_metadata=run_metadata, options=options)
                    else:
                        sess.run(compute)

                    if FLAGS.debug is True:
                        prof_timeline = timeline.Timeline(run_metadata.step_stats)
                        prof_ctf = prof_timeline.generate_chrome_trace_format()
                        timeline_saver.update_timeline(prof_ctf)

                if FLAGS.debug is True:
                    timeline_saver.save('./trace/worker-'+str(task_index)+'-test'+str(repeat)+'.json')

                sess.run(barrier)
    elif job_name == 'reducer' and (task_index == 0 or task_index == 1):
        _, outgoing = create_queue(num_workers)
        iteration = 1

        if FLAGS.debug is True:
            sol = np.load('./data/'+str(FLAGS.size)+'/sol.npy').astype(np.float32)
            truth_blocks = sol.reshape(int(sol.shape[0]/FLAGS.tile_size), FLAGS.tile_size, int(sol.shape[1]/FLAGS.tile_size), FLAGS.tile_size).swapaxes(1, 2).reshape(-1, FLAGS.tile_size, FLAGS.tile_size)
            sol = []

        config, cluster = create_cluster('reducer', FLAGS.num_reducers)
        server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, config=config, protocol=FLAGS.protocol)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=False,
                                               config=config) as sess:
            for repeat in range(FLAGS.num_tests):
                C_blocks = [np.zeros([FLAGS.tile_size, FLAGS.tile_size])] * num_tiles

                timeline_saver = TimeLiner()
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                sess.run(barrier)
                start_time = time.time()
                for c in range(int(num_blocks*num_blocks*num_blocks/FLAGS.dequeue_batch_size/2)):
                    if FLAGS.debug is True:
                        indexes, tiles = sess.run(outgoing[task_index], run_metadata=run_metadata, options=options)
                    else:
                        indexes, tiles = sess.run(outgoing[task_index])

                    for index, tile in zip(indexes, tiles):
                        i = int(index[0])
                        j = int(index[1])
                        C_blocks[i*num_blocks+j] = np.add(C_blocks[i*num_blocks+j], tile)

                        if FLAGS.debug is True:
                            prof_timeline = timeline.Timeline(run_metadata.step_stats)
                            prof_ctf = prof_timeline.generate_chrome_trace_format()
                            timeline_saver.update_timeline(prof_ctf)
                    #print('reduce '+str(c))
                end_time = time.time()

                if FLAGS.debug is True:
                    timeline_saver.save('./trace/reducer-'+str(task_index)+'-test'+str(repeat)+'.json')

                    if task_index == 0:
                        check = np.allclose(C_blocks[::2], truth_blocks[::2])
                        diff = np.subtract(C_blocks[::2], truth_blocks[::2])
                    else:
                        check = np.allclose(C_blocks[1::2], truth_blocks[1::2])
                        diff = np.subtract(C_blocks[1::2], truth_blocks[1::2]) 
                    print('correct: '+str(check)+' max err: '+str(np.amax(np.abs(diff.flatten()))))

                time_spent = (end_time - start_time)
                flops = ((N * N * N * 2 - N * N) / 10**9) / time_spent
                print('Test '+str(repeat)+' reducer: '+str(task_index)+': '+str(flops)+' Gflops/s '+str(time_spent)+' seconds')
                sess.run(barrier)
    else:
        config, cluster = create_cluster('reducer', FLAGS.num_reducers)
        server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, config=config, protocol=FLAGS.protocol)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=False,
                                               config=config) as sess:
            for test in range(FLAGS.num_tests):
                sess.run(barrier)
                sess.run(barrier)
