import time
import numpy as np
import dpmeans as dp

n = 1e5
k = 100

clustering = dp.Clustering(n)
clustering.randomize(k)

t = time.clock()
a = clustering.n_members
print(time.clock() - t)

t = time.clock()
b = clustering.array.sum(0)
print(time.clock() - t)

assert(np.all(a==b))
assert(k == clustering.k)

