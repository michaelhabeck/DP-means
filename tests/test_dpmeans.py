"""
Test DP-means for simple 2D data set
"""
import numpy as np
import dpmeans as dp
import matplotlib.pylab as plt

def generate_data(n_points, n_clusters, sigma=1., n_sigma=3, dim=2):
    """
    Random data
    """
    centers = np.random.standard_normal((n_clusters, dim)) * n_sigma * sigma
    weights = np.ones(n_clusters) / n_clusters
    counts  = np.random.multinomial(n_points, weights)
    limits  = np.add.accumulate(np.append(0, counts))
    
    data   = np.zeros((n_points, dim))
    labels = np.zeros(n_points)
    
    for k, (mu, n, a, b) in enumerate(zip(centers, counts, limits, limits[1:])):
        data[a:b]   = np.random.standard_normal((n,dim)) * sigma + mu
        labels[a:b] = k

    return data, labels, centers

n, k   = 1000, 3
cutoff = 5.

data, labels, centers = generate_data(n, k, n_sigma=5)

## compute loss of true clustering

dpmeans = dp.DPMeans(data, cutoff)
dpmeans.clusters.labels[...] = labels
dpmeans._centers[:dpmeans.clusters.k,:] = centers

loss_truth = dpmeans.loss()

## run DP-means

dpmeans = dp.DPMeans(data, cutoff)

loss = dpmeans.run(verbose=1)

fig, ax = plt.subplots(1,3,figsize=(12,4))
ax = list(ax.flat)

for i in range(k):
    ax[0].scatter(*data[labels==i].T)
for i in range(dpmeans.clusters.k):
    ax[1].scatter(*data[dpmeans.clusters.labels==i].T)
ax[2].plot(loss, lw=3)
ax[2].axhline(loss_truth, color='r', ls='--', lw=2)

fig.tight_layout()