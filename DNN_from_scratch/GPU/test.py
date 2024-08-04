import cupy as cp
import numpy as np
from time import time

a = cp.random.rand(1000)
b = np.random.rand(1000)
c = cp.random.rand(4000 * 1000).reshape(4000, 1000)
d = np.random.rand(4000 * 1000).reshape(4000, 1000)

start = time()
for i in range(1000):
    e = cp.dot(c,a)
end = time()
print(end - start)

start = time()
for i in range(1000):
    e = np.dot(d,b)
end = time()
print(end - start)