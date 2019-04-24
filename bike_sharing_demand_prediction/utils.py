#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy.sparse as sp
import pandas as pd
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import datetime


# # Model Architecture

# In[2]:


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




