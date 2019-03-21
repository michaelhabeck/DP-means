import numpy as np

class Clustering(object):
    """Clustering

    Class for storing cluster assignments. Provides some convenience
    functions to characterize the clustering. 
    """
    def __init__(self, n_items):
        if not (n_items > 0):
            msg = 'Number of items must be a positive integer'
            raise ValueError(msg)
        self.labels = np.zeros(int(n_items),dtype='i')

    @property
    def k(self):
        """
        Number of clusters
        """
        return self.labels.max() + 1

    @property
    def array(self):
        """
        Binary membership matrix
        """
        return np.equal.outer(self.labels, np.arange(self.k)).astype('i')

    @property
    def n_members(self):
        """
        Number of items assigned to each cluster
        """
        clusters, n_counts = np.unique(self.labels, return_counts=True)
        n_members = np.zeros(self.k,dtype='i')
        n_members[clusters] = n_counts
        return n_members

    @property
    def emtpy(self):
        """
        Return indices of empty clusters
        """
        return np.indices(self.n_members==0)[0]

    @property
    def nonemtpy(self):
        """
        Return indices of populated clusters
        """
        return np.indices(self.n_members>0)[0]

    def randomize(self, k=None):
        """
        Randomize cluster assignments
        """
        k = int(k) if k is not None else self.k
        self.labels[...] = np.random.randint(0, k, len(self.labels))

    
