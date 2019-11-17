import tensorflow as tf
import numpy as np

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    #adj[np.isnan(adj)] = 0.
    adj = tf.abs(adj)
    rowsum = tf.reduce_sum(adj, 1)# sum of each row

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



def masked_mae_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    #print (preds.shape)
    #print (labels.shape)
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)

def masked_mae_tf_by_horizon(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    preds_reshape = tf.reshape(preds, [-1, sn, horizon])
    labels_reshape = tf.reshape(labels, [-1, sn, horizon])
    
    res = []
    
    for i in range(horizon):
        labels = labels_reshape[:, :, 0:(i+1)]
        preds = preds_reshape[:, :, 0:(i+1)]
        
        if np.isnan(null_val):
            mask = ~tf.is_nan(labels)
        else:
            mask = tf.not_equal(labels, null_val)
        mask = tf.cast(mask, tf.float32)
        mask /= tf.reduce_mean(mask)
        mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
        loss = tf.abs(tf.subtract(preds, labels))
        loss = loss * mask
        loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
        
        res.append(tf.reduce_mean(loss))
        
    return res


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    #print (data.shape)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets,:,0] # only speed information, no hour of day
        #print (y_t.shape)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

