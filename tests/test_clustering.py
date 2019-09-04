import numpy as np
import dpmeans as dp

from timeit import default_timer as timer

n = 1e5
k = 100

clustering = dp.Clustering(n)
clustering.randomize(k)

start = timer()
a = clustering.n_members
end = timer()
print(end - start)

start = timer()
b = clustering.array.sum(0)
end = timer()
print(end - start)

assert(np.all(a==b))
assert(k == clustering.k)

