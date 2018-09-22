import os
import tensorflow as tf
import time
from hostlist import expand_hostlist

flags = tf.flags
flags.DEFINE_integer('iters', 10, 'Number of additions in each test')
flags.DEFINE_integer('runs', 10, 'Number of tests')
flags.DEFINE_integer('data_mb', 16, 'Size of vector in MBs')
flags.DEFINE_string('protocol', 'grpc', 'Protocol')
flags.DEFINE_string('device', 'gpu', 'Device to place vector on')
flags.DEFINE_string('show_placement', 'False', 'Show device placement')
FLAGS = flags.FLAGS

dtype=tf.float32
params_size = int(FLAGS.data_mb * 1024 * 1024 / 4.0)
sharded_params_size = params_size

# barrier
def build_barrier(num):
	with tf.variable_scope('barrier'):
		with tf.device('/job:ps/task:0/cpu:0'):
			counter = tf.get_variable(name='counter', shape=[], initializer=tf.zeros_initializer(), dtype=tf.int32, use_resource=True)
			increase_counter = tf.assign_add(counter, 1)
			c = lambda i: tf.not_equal(tf.mod(tf.add(i, num), num), 0)
			b = lambda i: counter
			with tf.control_dependencies([increase_counter]):
				barrier = tf.while_loop(c, b, [counter])
	return barrier

def create_worker(task_index, num_workers):
	# parameters for workers
	params = []
	with tf.variable_scope('ps-0'):
		with tf.device('job:ps/task:0/%s:0' % FLAGS.device):
			for i in range(num_workers):
				params.append(tf.get_variable(name='params-'+str(i), shape=[sharded_params_size], dtype=dtype, initializer=tf.zeros_initializer, use_resource=True))

	# vector to push to ps
	with tf.variable_scope('worker-%d' % task_index):
		with tf.device('job:worker/task:%d/%s:0' % (task_index, FLAGS.device)):
			update = tf.get_local_variable(name='update', shape=[sharded_params_size], dtype=dtype, initializer=tf.ones_initializer, use_resource=True)
			add_op = params[task_index].assign_add(update)

	return params[task_index], add_op

def main(argv):
	tf.logging.set_verbosity(tf.logging.INFO)
	#os.environ['CUDA_VISIBLE_DEVICE'] = ''
	if FLAGS.protocol == "grpc+mpi":
		rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
		num_tasks = int(os.environ['OMPI_COMM_WORLD_SIZE'])
	else:
		rank = int( os.environ['SLURM_PROCID'] )
		num_tasks = int( os.environ['SLURM_NTASKS'] )

	# get host list and assign jobs
	tf_hostlist = [ ('%s:22222' % host) for host in expand_hostlist( os.environ['SLURM_NODELIST']) ]  
	num_ps = 1
	num_workers = num_tasks - num_ps

	if rank < num_ps:
		job_name = 'ps'
		task_index = rank
	else:
		job_name = 'worker'
		task_index = rank - num_ps
		param, add_op = create_worker(task_index, num_workers)

	# barrier for workers + ps
	barrier = build_barrier(num_workers+1)

	cluster = tf.train.ClusterSpec({ 'ps': tf_hostlist[0:num_ps], 'worker': tf_hostlist[num_ps:] })
	server  = tf.train.Server(cluster.as_cluster_def(), job_name=job_name, task_index=task_index, protocol=FLAGS.protocol)
	config = tf.ConfigProto(log_device_placement=FLAGS.show_placement)
	print(cluster)

	if job_name == 'ps':
		# wait for graceful exit
		with tf.train.MonitoredTrainingSession(master=server.target,
							is_chief=False,
							config=config) as sess:
			sess.run(barrier)
			sess.run(barrier)
	else:
		with tf.train.MonitoredTrainingSession(master=server.target,
							is_chief=(task_index==0),
							config=config) as sess:
			sess.run(barrier)
			sess.run(add_op.op)

			for i in range(FLAGS.runs):
				start_time = time.time()
				for _ in range(FLAGS.iters):
					sess.run(add_op.op)
				end_time = time.time()
				elapsed_time = end_time - start_time
				rate = float(FLAGS.iters) * FLAGS.data_mb / elapsed_time
				print(str(i)+' worker '+str(task_index)+': start: '+str(start_time)+' end: '+str(end_time)+' rate: '+str(rate)+' MB/s')

			sess.run(barrier)

if __name__=='__main__':
	tf.app.run(main=main, argv=None)
