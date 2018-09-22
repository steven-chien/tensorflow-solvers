from tensorflow.python.client import timeline
import json
import math
import time
import numpy as np
import os
from hostlist import expand_hostlist
import tensorflow as tf

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

tf.app.flags.DEFINE_integer('size', 19, 'Size of vector as power of two.')
tf.app.flags.DEFINE_integer('num_tiles', 4, 'Number of tiles.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')
tf.app.flags.DEFINE_integer('num_reducers', 1, 'Number of reducers, at least one. Only the first reducer is used and the rest are idle.')
tf.app.flags.DEFINE_integer('num_tests', 6, 'Number of tests to repeat.')
tf.app.flags.DEFINE_integer('dequeue_batch_size', 2, 'Number of result tiles to collect by reducer in each dequeue.')
tf.app.flags.DEFINE_string('protocol', 'grpc+verbs', 'Transfer protocol.')
tf.app.flags.DEFINE_boolean('debug', False, 'Debug.')
FLAGS = tf.app.flags.FLAGS
vector_size = 2**FLAGS.size

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
    def map_func(target, tile_path):
        tile = np.load(tile_path.decode())
        return target.astype(np.float32), tile.astype(np.complex128)
 
    def generator():
        for i in range(FLAGS.num_tiles):
            yield i, './data/2_'+str(FLAGS.size)+'/'+str(FLAGS.num_tiles)+'/x-'+str(i)+'.npy'

    tile_size = int(N/FLAGS.num_tiles)
    dataset = tf.data.Dataset.from_generator(generator,
                                             (tf.float32, tf.string),
                                             (tf.TensorShape([]), tf.TensorShape([])))
    dataset = dataset.shard(num_workers, task_index)
    dataset = dataset.map(lambda target, tile_path: tuple(tf.py_func(map_func, [target, tile_path], [tf.float32, tf.complex128,])))
    dataset = dataset.prefetch(8)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
    iterator = dataset.make_initializable_iterator()

    return iterator

def create_queue(num_workers):
    tile_size = int(vector_size/FLAGS.num_tiles)

    with tf.device('/job:reducer/task:0'):
        incoming = tf.FIFOQueue(capacity=FLAGS.dequeue_batch_size*num_workers, dtypes=(tf.float32, tf.complex128), shapes=[[], [tile_size]], name='incoming', shared_name='incoming')
    outgoing = incoming.dequeue_many(FLAGS.dequeue_batch_size)

    return incoming, outgoing

def create_graph(task_index, num_workers):
    incoming, outgoing = create_queue(num_workers)
    iterator = build_dataset(task_index, num_workers)
    idx, seq = iterator.get_next()

    with tf.device('/job:worker/task:%d/gpu:0' % task_index):
        compute = tf.fft(seq)

    compute = incoming.enqueue((idx, compute))
    #with tf.control_dependencies([compute]):
    #    compute = tf.no_op()

    return compute, iterator.initializer

def pack_results(dft_results, n):
    """ Add dft results recursively

    """
    length = dft_results.shape[0]
    even = np.empty(length // 2, dtype=np.complex128)
    odd = np.empty(length // 2, dtype=np.complex128)
    for level in range(n-1, -1, -1):
        # Actually, half-size
        size = length // 2**level
        half_size = size // 2
        factor = np.exp(-2j*np.pi*np.arange(size) / size)
        for s in [slice(i, length, 2**level) for i in range(2**level)]:
            even[:half_size] = dft_results[s][::2] + factor[:half_size] * dft_results[s][1::2]
            odd[:half_size] = dft_results[s][::2] + factor[half_size:] * dft_results[s][1::2]
            dft_results[s][:half_size] = even[:half_size]
            dft_results[s][half_size:] = odd[:half_size]
    return dft_results

def create_cluster(job_name, task_index):
    rank = int( os.environ['SLURM_PROCID'] )
    num_gpus = FLAGS.num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank%num_gpus)

    tf_hostlist = []
    for host in expand_hostlist(os.environ['SLURM_NODELIST']):
        for gpu_id in range(num_gpus):
            tf_hostlist.append("%s:%d" % (host, 8888+gpu_id))

    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    cluster = tf.train.ClusterSpec({ "reducer": tf_hostlist[0:FLAGS.num_reducers], "worker": tf_hostlist[FLAGS.num_reducers:] })

    return config, cluster

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
#
    rank = int( os.environ['SLURM_PROCID'] )
    num_tasks = int(os.environ['SLURM_NTASKS'])
    num_workers = num_tasks - FLAGS.num_reducers

    if rank < FLAGS.num_reducers:
        job_name = 'reducer'
        task_index = rank
    else:
        job_name = 'worker'
        task_index = rank - FLAGS.num_reducers

    N = int(vector_size)

    barrier = build_barrier(num_tasks)
    compute, init_iterator = create_graph(task_index, num_workers)
    timeline_saver = TimeLiner()

    print('job: '+str(job_name)+' idx: '+str(task_index))
    if job_name == 'worker':
        config, cluster = create_cluster('worker', num_tasks-FLAGS.num_reducers)
        server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, config=config, protocol=FLAGS.protocol)
    
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index==0),
                                               config=config) as sess:
            for repeat in range(FLAGS.num_tests):
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                sess.run(init_iterator)

                sess.run(barrier)
                for i in range(int(FLAGS.num_tiles/num_workers)):
                    if FLAGS.debug is True:
                        sess.run(compute, run_metadata=run_metadata, options=options)
                        prof_timeline = timeline.Timeline(run_metadata.step_stats)
                        prof_ctf = prof_timeline.generate_chrome_trace_format()
                        timeline_saver.update_timeline(prof_ctf)
                        print('compute %d ...' % i)
                    else:
                        sess.run(compute)

                if FLAGS.debug is True:
                    timeline_saver.save('./trace/worker-'+str(task_index)+'-test'+str(repeat)+'.json')

                sess.run(barrier)
    elif job_name == 'reducer' and task_index == 0:
        _, outgoing = create_queue(num_workers)
        blocks = np.zeros([vector_size]).astype(np.complex128)
        result_tiles = [None] * FLAGS.num_tiles

        if FLAGS.debug is True:
            truth = np.load('./data/2_'+str(FLAGS.size)+'/'+str(FLAGS.num_tiles)+'/sol.npy')

        config, cluster = create_cluster('reducer', FLAGS.num_reducers)
        server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, config=config, protocol=FLAGS.protocol)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=False,
                                               config=config) as sess:
            for repeat in range(FLAGS.num_tests):
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                sess.run(barrier)
                start_time = time.time()
                for c in range(int(FLAGS.num_tiles/FLAGS.dequeue_batch_size)):
                    if FLAGS.debug is True:
                        indexes, tiles = sess.run(outgoing, run_metadata=run_metadata, options=options)
                        prof_timeline = timeline.Timeline(run_metadata.step_stats)
                        prof_ctf = prof_timeline.generate_chrome_trace_format()
                        timeline_saver.update_timeline(prof_ctf)
                        #print('reduce '+str(c))
                    else:
                        indexes, tiles = sess.run(outgoing)

                    for index, tile in zip(indexes, tiles):
                        i = int(index)
                        result_tiles[i] = tile
                end_time = time.time()
                print('finished fft: '+str(end_time - start_time))

                if FLAGS.debug is True:
                    for i in range(FLAGS.num_tiles):
                        blocks[i:vector_size:FLAGS.num_tiles] = result_tiles[i]
                    results = pack_results(blocks, int(math.log(FLAGS.num_tiles, 2)))
                    end_time = time.time()
                    time_spent = end_time - start_time
                    print('total time: '+str(time_spent))

                    check = np.allclose(truth, results)
                    print('correct: '+str(check))
                    timeline_saver.save('./trace/reducer-'+str(task_index)+'-test'+str(repeat)+'.json')

                sess.run(barrier)
    else:
        config, cluster = create_cluster('reducer', FLAGS.num_reducers)
        server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, config=config, protocol=FLAGS.protocol)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=False,
                                               config=config) as sess:
            for _ in range(FLAGS.num_tests):
                sess.run(barrier)
                sess.run(barrier)
