import numpy as np
import cupy as cp
from time import time


a = np.random.rand(100000000).reshape(10000, 10000)
s = time()
b = np.dot(a, a)
e = time()
print(e - s)


a = cp.random.rand(100000000).reshape(10000, 10000)
s = time()
b = cp.dot(a, a)
e = time()
print(e - s)