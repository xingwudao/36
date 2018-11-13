import numpy as np
import time

dim = 10000000
v1 = np.random.rand(dim)
v2 = np.random.rand(dim)

start = time.time()
v = 0.0
for i in range(dim):
    v += v1[i] * v2[i]
end = time.time()
cost = (end - start)*1000
print("v1 dot v2 = %.2f, cost %.2f ms" % (v,cost))

start = time.time()
v = np.dot(v1,v2)
end = time.time()
cost = (end - start)*1000
print("v1 dot v2 = %.2f, cost %.2f ms" % (v,cost))
