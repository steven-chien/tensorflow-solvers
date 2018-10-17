# CG solver

This launches a distributed CG solver on prepared input data.

## Parameters
- ```--size``` Size of  matrix A.
- ```--num_gpus``` How many GPUs per node, should be same as the number of processes launched on each node.
- ```--num_reducers``` How reducers. Currently only first is used.
- ```--iters``` How solver iterations.
- ```--checkpoint_steps``` Steps between checkpoint. Checkpoint disabled if not set.
- ```--protocol``` Communication protocol. Default is grpc+verbs.

## Input

```
cd data/
python generate_sparse.py [problem size] [list of number of workers, separated by comma]
```
A folder with the naming being the problem size will be created with sub-folders 2_workers, 4_workers, 8_workers, 16_workers with A-N.npy, where N is the portion resposible by process N over the total number of workers. delta.npy for initial delta and r.npy for initial residual are saved directly on in the folder without splitting. Global p and r will be initialized on the parameter server/reducer. Local p and r on workers will be initialized by the initialized value of r with respective portion. A is too large to be store in the graph and is initialized through a placeholder locally on workers. The initial solution is all zero and the problem is formulated in such a way that the final solution converges to one period of a sin function.

## Example
```
salloc --constraint="Haswell" --nodes=3 -t 00:15:00 -A <proj-name> --ntasks-per-node=1 --gres=gpu:K420:1 

srun -n 3 --kill-on-bad-exit=1 --unbuffered python cg.py --size=8192 --num_gpus=1 --num_ps=1 --num_iters=500 --protocol=grpc+verbs --checkpoint_steps=100

OR

mpirun -np 3 python -u cg.py --size=8192 --num_gpus=1 --num_ps=1 --num_iters=500 --protocol=grpc+mpi --checkpoint_steps=100
```

The solver requires a number of tasks that is equal to the number of workers + number of ps. For 2 workers, 3 tasks have to be launched and so on. If there are more than one GPU per node the number of GPUs per node can be specified and the value should be the same as --ntasks-per-node so that that specific number of tasks will be launched on each node. Each process will see one assigned GPU. A folder called checkpoint will be used to store checkpoints.

## Note
- Need better way of storing and loading large problems as sparse matrix instead of dense.
