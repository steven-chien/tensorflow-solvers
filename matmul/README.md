# CG solver

This launches a distributed CG solver on prepared input data.

## Parameters
- ```--size``` Size of matrices.
- ```--tile_size``` Size of tiles.
- ```--num_gpus``` How many GPUs per node, should be same as the number of processes launched on each node.
- ```--num_reducers``` How many reducers to use, must be at least two. Currently only the first and second are used.
- ```--dequeue_batch_size``` Number of tiles a reducer collects per dequeue, default to 16.
- ```--num_tests``` Number of tests to repeat.
- ```--protocol``` Communication protocol. Default is grpc+verbs, optionally grpc, grpc+mpi.
- ```--debug``` Turn on debug features which saves execution timeline in JSON and check computation answer.

## Input

```
cd data/
python generate.py [problem size] [list of tile sizes to split into, separated by comma]
```
A folder with the name being problem size will be created with sub-folders with names being the tile sizes. A and B matrices will be randomly generated and named as ```A_i_j.npy``` and ```B_i_j.npy``` where i and j represents the position of the tiles in the whole matrix. Solution is saved as ```sol.npy```.

## Example
```
salloc --constraint="Haswell" --nodes=4 -t 00:15:00 -A <proj-name> --ntasks-per-node=1 --gres=gpu:K420:1 

srun --kill-on-bad-exit=1 -l --unbuffered python -u matmul.py --size=32768 --tile_size=4096 --dequeue_batch_size=16 --num_gpus=1 --num_reducers=2 --num_tests=5 --debug=True
```

The application requires a number of tasks that is equal to the number of workers + number of reducers. Only the first two reducers are used and the rest will be idle. The two reducers receive tiles with target index being an odd number or an even number. The fequency of dequeue can be adjusted with ```--dequeue_batch_size```. The reducer extracts the resulting tiles from the queue into the Python Session and perform addition with Numpy.
