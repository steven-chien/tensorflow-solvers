# Bandwidth Estimator

This launches a Python script to perform addition of vector by sending a vector on one worker to a parameter server for addition and assignment.

## Parameters
- ```--runs``` Specify the number of tests to run.
- ```--iters``` Specify number of transfers to run for in integer in each test.
- ```--data_mb``` Specify size for each transfer in MB in integer.
- ```--protocol``` Specify protocol to use: grpc, grpc+verbs or grpc+mpi.
- ```--device``` Specify device to place the vector, cpu or gpu.

## Returns
Two or more processes will be launched on two or more nodes, and vectors of integers with size equivalent to specification will be created, one on worker and one on parameter server. An assign_add operation is created to push the vector on worker to parameter server. Number of run is specified by iterations. An average bandwidth in MB/s is reported. ( (transfer size * iterations) / time elapsed ).

## Example

```
salloc --nodes=2 -t 00:15:00 -A some-project --ntasks-per-node=1 --gres=gpu:K420:1
srun --kill-on-bad-exit=1 -l --unbuffered python bandwidth.py --runs=10 --iters=100 --data_mb=128 --protocol=grpc+verbs --device=gpu

OR

mpirun --tag-output python -u bandwidth.py --runs=10 --iters=100 --data_mb=128 --protocol=grpc+mpi --device=gpu
```

## Note
- MPI_OPTIMAL_PATH environment variable used by the MPI module is disabled by default. Tensors will be serialized before transfer instead of through RDMA. The reason is that execution will fail if Tensors resides on GPU and the MPI runtime used is not CUDA-aware.
- Execution add operation with .op to not fetch value from graph for avoiding transfer.
