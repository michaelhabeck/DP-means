"""
DP-means clustering
"""
import numpy as np

from .clustering import Clustering

from scipy.ndimage.measurements import sum as aggregate

from sklearn.neighbors import BallTree

class DPMeans(object):

    batch_size = 1000

    eps = 1e-100

    @property
    def cutoff(self):
        """
        Distance cutoff
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        """
        Set distance cutoff
        """
        if not value > 0.:
            msg = 'Distance cutoff (aka cluster penalty) must' + \
                  'be a positive float'
            raise ValueError(msg)
        
        self._cutoff  = float(value)
        self._penalty = self._cutoff**2

    @property
    def penalty(self):
        """
        Lambda parameter
        """
        return self._penalty

    @property
    def centers(self):
        """
        Cluster centers
        """
        return self._centers[:self.clusters.k]

    def __init__(self, data, cutoff, weights=None):
        """
        Arguments
        ---------
        
        data : rank-2 numpy array
          data matrix where each row corresponds to a multi-
          dimensional data vector
          
        cutoff : positive float
          distance cutoff or cluster penalty parameter

        weights : rank-1 numpy array
          optional weights associated with each data point
        """
        if data.ndim != 2:
            msg = 'Data matrix must be a rank-2 array'
            raise ValueError(msg)

        weights = np.ones(len(data)) if weights is None else weights

        if weights.ndim != 1 or len(weights) != len(data):
            msg = 'Data weights must be a rank-1 array of length {0}'
            raise ValueError(msg.format(len(data)))

        self.data     = data
        self.weights  = weights
        self.cutoff   = cutoff

        self.clusters = Clustering(len(self.data))

        self._centers = None
        self._add_new_batch()
        self._centers[0] = np.mean(self.data,0)

    def __str__(self):
        return '{0}(n_clusters={1})'.format(
            self.__class__.__name__, self.clusters.k)

    def _add_new_batch(self):
        """
        Private method that augments the array holding the cluster
        centers by a certain number of rows
        """
        batch = np.zeros((self.batch_size, self.data.shape[1]))
        if self._centers is None:
            self._centers = batch
        else:
            self._centers = np.concatenate([self._centers, batch],0)

    def get_unassigned(self):
        return np.random.permutation(len(self.data))

    def assign_point(self, index):
        """
        Assigns a data point (row in data matrix) to the closest
        cluster. If distance exceeds the cutoff, a new cluster is
        spawned. 
        """
        point = self.data[index]
        dist  = np.sum((point - self.centers)**2,1)

        if dist.min() <= self.penalty:
            self.clusters.labels[index] = dist.argmin()
            return

        ## create new cluster

        k = self.clusters.k
        if k >= len(self._centers):
            self._add_new_batch()

        self._centers[k] = point            
        self.clusters.labels[index] = k

    def remove_empty(self):
        """
        Remove empty clusters
        """
        nonempty   = self.clusters.nonempty        
        membership = self.clusters.array[:,nonempty]

        self._centers[:len(nonempty)] = self.centers[nonempty]
        self.clusters.labels[...] = membership.argmax(1)

    def update_centers(self):
        """
        Update cluster centers by averaging positions of members
        """
        clusters = np.arange(self.clusters.k)
        labels   = self.clusters.labels
        
        weights  = aggregate(self.weights, labels, index=clusters)

        centers  = np.zeros((self.data.shape[1], len(clusters)))
        for d in xrange(self.data.shape[1]):
            values = self.weights * self.data[:,d]
            centers[d,...] = aggregate(values, labels, index=clusters)
        centers /= (weights + self.eps)

        self._centers[:len(clusters)] = centers.T

    def next(self):
        """
        A single DP-means iteration that sweeps over all data points
        in some random order
        """
        unassigned = self.get_unassigned()
        for i in unassigned:
            self.assign_point(i)

        self.update_centers()

    def fitness(self):
        """
        Evaluates how well the clusters approximate the data
        """
        d = np.sum((self.data-self.centers[self.clusters.labels])**2,1)
        f = np.dot(self.weights, d) / (self.weights.sum() + self.eps)

        return len(self.data) * f

    def loss(self):
        """
        Evaluate loss function optimized by DP-means
        """
        return self.clusters.k * self.penalty + self.fitness()

    def stop(self, loss, tol):
        """
        Stop critertion : no significant change in loss function
        """
        if tol is None:
            return False
        
        if len(loss) >= 2:
            x, y = loss[-2:]
            return abs(x-y) / abs(x+y) < tol

        return False

    def run(self, n_iter=100, tol=1e-5, verbose=0):
        """
        Run DP-means iterations until loss function does no longer
        change significantly (specified by tolerance) or maximum
        number of iterations is reached.

        Arguments
        ---------
        n_iter : positive integer
          maximum number of iterations

        tol : float or None
          tolerance for checking local convergence

        verbose : non-negative integer
          specifies frequency with which progress of DP-means is
          reported (verbose=0 means no messages are shown, default)
        """
        loss = []
        i = 0

        while i < n_iter:

            next(self)
            loss.append(self.loss())
            
            if verbose and not i % verbose:
                print i, self.cutoff, self.clusters.k, loss[-1]

            if self.stop(loss, tol):
                break

            i += 1

        return loss

class FastDPMeans(DPMeans):
    """
    Faster implementation using a BallTree
    """
    debug = not True

    def get_unassigned(self):
        """
        Use BallTree to find nearest clusters
        """
        k = self.clusters.k

        if k == 1:
            return super(FastDPMeans, self).get_unassigned()

        tree = BallTree(self.centers, leaf_size=k+1)

        neigh, _ = tree.query_radius(self.data, self.cutoff, sort_results=True,
                                     return_distance=True)

        n_neigh    = np.array(map(len, neigh))
        assigned   = np.nonzero(n_neigh>0)[0]
        unassigned = np.nonzero(n_neigh==0)[0]

        self.clusters.labels[assigned] = [neigh[i][0] for i in assigned]

        return unassigned

