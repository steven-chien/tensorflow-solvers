import os
import sys
import numpy as np
from scipy.sparse import diags

if len(sys.argv) != 3:
    print('Enter problem N and number of workers.')
    sys.exit(1)

N = int(sys.argv[1])
workers = list(map(int, sys.argv[2].split(',')))

A = diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)).toarray()
print(A)
b = np.ones(N)
for i in range(N):
    b[i] = np.sin(2*np.pi*(i/N)) * (2*np.pi/N)*(2*np.pi/N)
b = b.reshape([N, 1])
print(b)

x = np.zeros([A.shape[0], 1])

r = np.subtract(b, np.matmul(A, x))
delta = np.dot(r.T, r)
print(delta)

if not os.path.exists('./'+str(N)):
    os.makedirs('./'+str(N))

#np.save('./'+str(N)+'/A.npy', A)
np.save('./'+str(N)+'/r.npy', r)
np.save('./'+str(N)+'/b.npy', b)
np.save('./'+str(N)+'/delta.npy', delta)

for num_workers in workers:
    for i in range(num_workers):
        if not os.path.exists('./'+str(N)+'/'+str(num_workers)+'_workers'):
            os.makedirs('./'+str(N)+'/'+str(num_workers)+'_workers')
        np.save('./'+str(N)+'/'+str(num_workers)+'_workers/A-'+str(i)+'.npy', A[int(i*(N/num_workers)):int((i+1)*(N/num_workers))])
        np.save('./'+str(N)+'/'+str(num_workers)+'_workers/b-'+str(i)+'.npy', b[int(i*(N/num_workers)):int((i+1)*(N/num_workers))])
