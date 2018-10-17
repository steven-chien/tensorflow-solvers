# CG solver

This launches a distributed CG solver on prepared input data.

## Parameters
- ```--size``` Size of vector as power of two.
- ```--num_tiles``` Number of tiles the vector is in.
- ```--num_gpus``` How many GPUs per node, should be same as the number of processes launched on each node.
- ```--num_reducers``` Number of reducers, at least one. Only the first reducer is used and the rest are idle.
- ```--num_tests``` Number of tests to repeat.
- ```--dequeue_batch_size``` Number of result tiles to collect by reducer in each dequeue.
- ```--protocol``` Transfer protocol, default is grpc+verbs, can be grpc or grpc+mpi
- ```--debug``` Perform merging, obtain total timing and save execution metadata as timeline.

## Input

```
cd data/
python generate.py [problem size] [num of tiles]
```
A folder with the name being problem size will be created with sub-folders with names being the number of interleaving tiles. Tiles will be named ```x-i.npy``` where i is the index of the tile. Solution is saved as ```sol.npy```.

## Example
```
salloc --constraint="Haswell" --nodes=3 -t 00:15:00 -A <proj-name> --ntasks-per-node=1 --gres=gpu:K420:1 

srun --kill-on-bad-exit=1 -l --unbuffered python -u fft.py --size=29 --tile_size=64 --dequeue_batch_size=16 --num_gpus=1 --num_reducers=1 --num_tests=5 --debug=True
```

The application requires a number of tasks that is equal to the number of workers + number of reducers. Only the first reducers are used and the rest will be idle. Workers individually load tiles and perform FFT and push results into the queue at merger. The merger extracts result times and resemble them into an array according to the index. Finally the computation of factors and merging is performed.
