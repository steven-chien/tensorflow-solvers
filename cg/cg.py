import os
from hostlist import expand_hostlist
import numpy as np
import tensorflow as tf
import time

tf.app.flags.DEFINE_integer('size', 8192, 'Size of matrix.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')
tf.app.flags.DEFINE_integer('num_reducers', 1, 'Number of reducers, minimal one. Only the first reducer performs computation and the rests are idle.')
tf.app.flags.DEFINE_integer('iters', 100, 'How solver iterations.')
tf.app.flags.DEFINE_integer('checkpoint_steps', -1, 'Stereducer between checkpoint.')
tf.app.flags.DEFINE_string('protocol', 'grpc+verbs', 'Transfer protocol.')
FLAGS = tf.app.flags.FLAGS

# barrier
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

# global parameters in solver
def build_parameters(N, num_worker):
    size_per_task = int(N/num_worker)
    global_step = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(-1), trainable=False, dtype=tf.int32)

    with tf.variable_scope('reducer-0'):
        with tf.device('/job:reducer/task:0/cpu:0'):
            reducer_alpha = tf.get_variable(name='alpha', shape=[], initializer=tf.zeros_initializer(), dtype=tf.float64, use_resource=True)
            reducer_beta = tf.get_variable(name='beta', shape=[], initializer=tf.zeros_initializer(), dtype=tf.float64, use_resource=True)
            reducer_delta_new = tf.get_variable(name='delta_new', initializer=np.load('data/'+str(N)+'/delta.npy').astype(np.float64)[0][0], dtype=tf.float64, use_resource=True)
            reducer_delta_old = tf.get_variable(name='delta_old', initializer=reducer_delta_new.initialized_value(), dtype=tf.float64, use_resource=True)
            reducer_p = tf.get_variable(name='p', initializer=np.load('data/'+str(N)+'/r.npy').astype(np.float64).flatten(), dtype=tf.float64, use_resource=True)
            reducer_x = tf.get_variable(name='x', initializer=tf.zeros_initializer(), shape=[N], dtype=tf.float64, use_resource=True)
            residual_norm = tf.get_variable(name='r_norm', dtype=tf.float64, initializer=tf.zeros_initializer(), shape=[], use_resource=True)

    return reducer_alpha, reducer_beta, reducer_delta_new, reducer_delta_old, reducer_p, reducer_x, residual_norm, global_step

# data and result queue to communicate between worker and parameter server
def build_queues(N, num_worker):
    size_per_task = int(N/num_worker)

    with tf.variable_scope('reducer-0'):
        with tf.device('/job:reducer/task:0/cpu:0'):
            partial_alpha_queue = tf.FIFOQueue(capacity=num_worker, dtypes=tf.float64, shapes=[], shared_name='partial_alpha_shared_queue')
            alpha_queue = tf.FIFOQueue(capacity=num_worker, dtypes=tf.float64, shapes=[], shared_name='alpha_shared_queue')
            partial_delta_queue = tf.FIFOQueue(capacity=num_worker, dtypes=tf.float64, shapes=[], shared_name='partial_delta_shared_queue')
            beta_queue = tf.FIFOQueue(capacity=num_worker, dtypes=tf.float64, shapes=[], shared_name='beta_shared_queue')
            p_queue = tf.PriorityQueue(capacity=num_worker, types=(tf.float64, tf.float64), shapes=[[size_per_task, 1], [size_per_task, 1]], shared_name='p_shared_queue')
            p_result_queue = tf.FIFOQueue(capacity=num_worker, dtypes=(tf.float64), shapes=[N, 1], shared_name='p_result_shared_queue')

    return partial_alpha_queue, alpha_queue, partial_delta_queue, beta_queue, p_queue, p_result_queue

# build parameter and queues on PS and reducers for the solver
def build_reducer(N, num_worker):
    size_per_task = int(N/num_worker)
    reducer_alpha, reducer_beta, reducer_delta_new, reducer_delta_old, reducer_p, reducer_x, residual_norm, global_step = build_parameters(N, num_worker)
    partial_alpha_queue, alpha_queue, partial_delta_queue, beta_queue, p_queue, p_result_queue = build_queues(N, num_worker)
    increase_step = tf.assign_add(global_step, 1, use_locking=True)

    with tf.variable_scope('reducer-0'):
        with tf.device('/job:reducer/task:%d/cpu:0' % 0):

            # compute new alpha and distribute to workers
            partial_alphas = partial_alpha_queue.dequeue_many(num_worker)
            compute_local_alpha = tf.assign(reducer_alpha, tf.divide(reducer_delta_new, tf.reduce_sum(partial_alphas)))

            with tf.control_dependencies([compute_local_alpha]):
                enqueue_alpha = alpha_queue.enqueue_many((tf.fill([num_worker], reducer_alpha)))
        
                # compute delta and beta and distribute
                with tf.control_dependencies([enqueue_alpha]):
                    partial_deltas = partial_delta_queue.dequeue_many(num_worker)
                    update_delta_new = tf.assign(reducer_delta_new, tf.reduce_sum(partial_deltas))

                    update_beta = tf.assign(reducer_beta, tf.divide(update_delta_new, reducer_delta_old))
                    enqueue_beta = beta_queue.enqueue_many((tf.fill([num_worker], update_beta)))
        
                    # merge updated p and x and distribute
                    with tf.control_dependencies([enqueue_beta]):
                        compute_norm = tf.assign(residual_norm, tf.sqrt(reducer_delta_new))

                        partial_p = p_queue.dequeue_many(num_worker)
                        merged_p = tf.reshape(partial_p[2], shape=[N])
                        update_reducer_p = tf.assign(reducer_p, merged_p)
                        enqueue_p = p_result_queue.enqueue_many((tf.reshape(tf.tile(update_reducer_p, [num_worker]), shape=[num_worker, N, 1])))

                        merge_x = tf.reshape(partial_p[1], shape=[N])
                        update_reducer_x = tf.assign(reducer_x, merge_x)
                        update_delta_old = tf.assign(reducer_delta_old, reducer_delta_new)

                        # increase step counter
                        with tf.control_dependencies([update_reducer_x, enqueue_p, update_delta_old, compute_norm]):
                            step_reducer = [reducer_alpha.read_value(), reducer_beta.read_value(), reducer_delta_new.read_value(), reducer_delta_old.read_value(), residual_norm.read_value(), increase_step]
    return step_reducer, reducer_x

def build_worker(N, num_worker, task_index):
    size_per_task = int(N/num_worker)
    idx_start = int(task_index*size_per_task)
    idx_end = int((task_index+1)*size_per_task)

    reducer_alpha, _, _, _, reducer_p, reducer_x, _, _ = build_parameters(N, num_worker)
    partial_alpha_queue, alpha_queue, partial_delta_queue, beta_queue, p_queue, p_result_queue = build_queues(N, num_worker)

    with tf.variable_scope('task-'+str(task_index)):
        with tf.device('/job:worker/task:%d/cpu:0' % task_index):
            # A is too large to store in graph, must initialize from session and placeholder
            A_local_placeholder = tf.placeholder(dtype=tf.float64, shape=[size_per_task, N])
            b_local = tf.get_local_variable(name='b', initializer=np.load('data/'+str(N)+'/'+str(num_worker)+'_workers/b-'+str(task_index)+'.npy'), dtype=tf.float64, use_resource=True)

        with tf.device('/job:worker/task:%d/gpu:0' % task_index):
            A_local = tf.get_local_variable(name='A_local',    shape=[size_per_task, N], dtype=tf.float64, use_resource=True)

            x_local = tf.get_local_variable(name='x_local',
                            initializer=tf.reshape(tf.gather(reducer_x.initialized_value(), tf.range(idx_start, idx_end)), [size_per_task, 1]),
                            dtype=tf.float64,
                            use_resource=True)

            r_local = tf.get_local_variable(name='r_local',
                            initializer=tf.reshape(tf.gather(reducer_p.initialized_value(), tf.range(idx_start, idx_end)), [size_per_task, 1]),
                            dtype=tf.float64,
                            use_resource=True)

            #alpha = tf.get_local_variable(name='alpha', initializer=reducer_alpha.initialized_value(), dtype=tf.float64, use_resource=True)
            p = tf.get_local_variable(name='p', initializer=tf.reshape(reducer_p.initialized_value(), [N, 1]), dtype=tf.float64,    use_resource=True)
            p_local = tf.gather(p, tf.range(idx_start, idx_end))
            q_local = tf.get_local_variable(name='q_local', shape=[size_per_task, 1], initializer=tf.zeros_initializer(), dtype=tf.float64, use_resource=True)

            init_a = tf.assign(A_local, A_local_placeholder)
            init_r = tf.assign(r_local, tf.subtract(b_local.initialized_value(), tf.matmul(init_a, tf.reshape(reducer_x.initialized_value(), [N, 1]))))

            # compute q local
            compute_q_local = tf.assign(q_local, tf.matmul(A_local, p))
            new_alpha = tf.reduce_sum(tf.multiply(p_local, compute_q_local))
            compute_partial_alpha = partial_alpha_queue.enqueue(new_alpha)

            with tf.control_dependencies([compute_partial_alpha]):
                # compute partial alpha, send to reducer and extract reduced result
                alpha = alpha_queue.dequeue()

                # update x and r with updated alpha and send to reducer
                compute_x_local = tf.assign_add(x_local, tf.scalar_mul(alpha, p_local))
                compute_r_local = tf.assign_add(r_local, tf.scalar_mul(tf.negative(alpha), q_local))

                #with tf.control_dependencies([compute_r_local, compute_x_local]):
                # compute partial delta and send to reducer and obtain new beta
                compute_partial_delta = partial_delta_queue.enqueue(tf.reduce_sum(tf.matmul(compute_r_local, compute_r_local, transpose_a=True)))

                with tf.control_dependencies([compute_partial_delta]):
                    # obtain new beta and compute updated p and send to reducer
                    beta = beta_queue.dequeue()
                    compute_p_local = p_queue.enqueue((task_index, compute_x_local, tf.add(tf.scalar_mul(beta, p_local), r_local)))

                    with tf.control_dependencies([compute_p_local]):
                        # update p with merged p from reducer
                        worker_step = tf.assign(p, p_result_queue.dequeue())

    return A_local_placeholder, init_r, worker_step

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
    server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, config=config, protocol=FLAGS.protocol)

    return config, cluster, server

def run_reducer(task_index, chief_only_hooks, hooks):
    job_name = "reducer"
    num_worker = int(os.environ['SLURM_NTASKS']) - FLAGS.num_reducers

    # number of workers + 1 reducer
    barrier = build_barrier(num_worker+1)
    step_reducer, reducer_x = build_reducer(FLAGS.size, num_worker)

    config, cluster, server = create_cluster(job_name, task_index)
    with tf.train.MonitoredTrainingSession(master=server.target,
                        is_chief=(task_index==0),
                        chief_only_hooks=chief_only_hooks,
                        hooks=hooks,
                        checkpoint_dir='./checkpoints',
                        save_checkpoint_steps=FLAGS.checkpoint_steps,
                        config=config) as sess:
        # wait for all processes to be ready
        sess.run(barrier)
        start_time = time.time()
        print('start time '+str(start_time))

        while not sess.should_stop():
            local_alpha, local_beta, local_delta, local_delta_old, local_residual, step = sess.run(step_reducer)
            print('step={:<10d}alpha={:<20g}delta={:<20g}beta={:<20g}residual={:<20g}'.format(step, local_alpha, local_delta, local_beta, local_residual))

        end_time = time.time()
        print('end time '+str(end_time))
        print('runtime: '+str(end_time-start_time))

def run_worker(task_index, chief_only_hooks, hooks):
    job_name = "worker"
    num_worker = int(os.environ['SLURM_NTASKS']) - FLAGS.num_reducers

    matrix_a = np.load('data/'+str(FLAGS.size)+'/'+str(num_worker)+'_workers/A-'+str(task_index)+'.npy')
    A_local_placeholder, init_r, worker_step = build_worker(FLAGS.size, num_worker, task_index)

    # number of workers + 1 reducer
    barrier = build_barrier(num_worker+1)

    config, cluster, server = create_cluster(job_name, task_index)
    with tf.train.MonitoredTrainingSession(master=server.target,
                        is_chief=False,
                        chief_only_hooks=chief_only_hooks,
                        hooks=hooks,
                        config=config) as sess:

        # initialize A
        sess.run([init_r], feed_dict={ A_local_placeholder: matrix_a })
        sess.run(barrier)
        while not sess.should_stop():
            sess.run(worker_step)

def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    # assign task to process
    rank = int( os.environ['SLURM_PROCID'] )
    num_tasks = int( os.environ['SLURM_NTASKS'] )

    if rank < FLAGS.num_reducers:
        job_name = "reducer"
        task_index = rank
    else:
        job_name = "worker"
        task_index = rank - FLAGS.num_reducers

    # first step is 0
    stop_hook = tf.train.StopAtStepHook(last_step=FLAGS.iters)
    hooks = [stop_hook]
    chief_only_hooks = []

    g = tf.get_default_graph()
    with g.as_default():
        if job_name == "reducer" and task_index != 0:
            # for now only one reducer is used
            _, _, server = create_cluster(job_name, task_index)
            server.join()
        elif job_name == "reducer":
            run_reducer(task_index, chief_only_hooks, hooks)
        else:
            run_worker(task_index, chief_only_hooks, hooks)

if __name__ == "__main__":
    tf.app.run(main=main, argv=None)
