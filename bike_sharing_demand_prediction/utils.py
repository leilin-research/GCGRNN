import tensorflow as tf

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    #adj[np.isnan(adj)] = 0.
    adj = tf.abs(adj)
    rowsum = tf.reduce_sum(adj, 1)# sum by row

    d_inv_sqrt = tf.pow(rowsum, -0.5)
   
    #d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    d_mat_inv_sqrt = tf.diag(d_inv_sqrt)

    return tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean




