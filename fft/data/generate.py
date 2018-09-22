import numpy as np
import os
import sys

if int(len(sys.argv)) != 3:
    print('Enter vector size in power of two and number of tiles')

size = 2**int(sys.argv[1])
num_tiles = int(sys.argv[2])
tile_size = size//num_tiles
print('num tiles: '+str(num_tiles))
print('tile size: '+str(tile_size))

if not os.path.exists('./2_'+str(sys.argv[1])+'/'+str(num_tiles)):
    os.makedirs('./2_'+str(sys.argv[1])+'/'+str(num_tiles))

x = np.random.random([size]).astype(np.complex128)
sol = np.fft.fft(x).astype(np.complex128)

for i in range(num_tiles):
    np.save('./2_'+str(sys.argv[1])+'/'+str(num_tiles)+'/x-'+str(i)+'.npy', x[i:size:num_tiles])

np.save('./2_'+str(sys.argv[1])+'/'+str(num_tiles)+'/sol.npy', sol)
