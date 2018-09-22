import numpy as np
import os
import sys

N = 256
sizes = '32,64,128'

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    sizes = sys.argv[2]

sizes = list(map(int, sizes.split(',')))

if not os.path.exists('./'+str(N)):
    os.makedirs('./'+str(N))

A = np.random.random([N,N]).astype(np.float32)
B = np.random.random([N,N]).astype(np.float32)
np.save('./'+str(N)+'/A.npy', A)
np.save('./'+str(N)+'/B.npy', B)
sol = np.matmul(A, B)
np.save('./'+str(N)+'/sol.npy', sol)
sol = []

for tile_size in sizes:
    num_blocks = int(N/tile_size)

    if not os.path.exists('./'+str(N)+'/'+str(tile_size)):
        os.makedirs('./'+str(N)+'/'+str(tile_size))

    A_blocks = A.reshape(int(A.shape[0]/tile_size), tile_size, int(A.shape[1]/tile_size), tile_size).swapaxes(1, 2).reshape(-1, tile_size, tile_size)
    for i in range(num_blocks):
        for j in range(num_blocks):
            np.save('./'+str(N)+'/'+str(tile_size)+'/A_'+str(i)+'_'+str(j)+'.npy', A_blocks[i*num_blocks+j])
    A_blocks = []
    
    B_blocks = B.reshape(int(B.shape[0]/tile_size), tile_size, int(B.shape[1]/tile_size), tile_size).swapaxes(1, 2).reshape(-1, tile_size, tile_size)
    for i in range(num_blocks):
        for j in range(num_blocks):
            np.save('./'+str(N)+'/'+str(tile_size)+'/B_'+str(i)+'_'+str(j)+'.npy', B_blocks[i*num_blocks+j])

    B_blocks = []
