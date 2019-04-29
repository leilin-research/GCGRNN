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

from utils import normalize_adj, StandardScaler, generate_graph_seq2seq_io_data
from gcn_hourofday import gcn, gcnn_ddgf_mae

# Import Data

raw_data = pd.read_hdf('data/metr-la.h5')


# Split Data into Training, Validation and Testing

node_num = 207 # node number 
feature_in = 12 # number of features at each node, e.g., bike sharing demand from past 24 hours
horizon = 12 # the length to predict, e.g., predict the future one hour bike sharing demand

x_offsets = np.sort(
    # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
    np.concatenate((np.arange(-feature_in+1, 1, 1),))
)
# Predict the next one hour
y_offsets = np.sort(np.arange(1, 1+ horizon, 1))

X_whole, Y_whole = generate_graph_seq2seq_io_data(
    raw_data,
    x_offsets=x_offsets,
    y_offsets=y_offsets,
    add_time_in_day=True,
    add_day_in_week=False,
)

X_whole = np.moveaxis(X_whole, 2, 1) # from (34249, 12, 207, 2) to (34249, 207, 12, 2)
Y_whole = np.moveaxis(Y_whole, 2, 1) # from (34249, 12, 207) to (34249, 207, 12)

print (X_whole.shape) 

num_samples = X_whole.shape[0]
Y_whole = np.reshape(Y_whole, [num_samples, -1])

num_test = round(num_samples * 0.2)
num_train = round(num_samples * 0.7)
num_val = num_samples - num_test - num_train

X_training = X_whole[:num_train, :]
Y_training = Y_whole[:num_train, :]

# shuffle the training dataset
perm = np.arange(X_training.shape[0])
np.random.shuffle(perm)
X_training = X_training[perm]
Y_training = Y_training[perm]

X_val = X_whole[num_train:num_train+num_val, :]
Y_val = Y_whole[num_train:num_train+num_val, :]

X_test = X_whole[num_train+num_val:num_train+num_val+num_test, :]
Y_test = Y_whole[num_train+num_val:num_train+num_val+num_test, :]


scaler = StandardScaler(mean=X_training[..., 0].mean(), std=X_training[..., 0].std())

X_training[..., 0] = scaler.transform(X_training[..., 0])
print (X_training.shape) 
Y_training = scaler.transform(Y_training)

X_val[..., 0] = scaler.transform(X_val[..., 0])
Y_val = scaler.transform(Y_val)

X_test[..., 0] = scaler.transform(X_test[..., 0])
Y_test = scaler.transform(Y_test)

# Hyperparameters

learning_rate = 0.01 # learning rate
decay = 0.9
batchsize = 100 # batch size 

hidden_num_layer = [10, 10, 20] # determine the number of hidden layers and the vector length at each node of each hidden layer
hidden_num_layer1 = [5, 5] #[5, 2, 2]
keep = 1 # drop out probability

early_stop_th = 500 # early stopping threshold, if validation RMSE not dropping in continuous 20 steps, break
training_epochs = 1000 # total training epochs


# Training

for i in range(10):

     start_time = datetime.datetime.now()

     val_error, predic_res, test_Y, test_error = gcnn_ddgf_mae(hidden_num_layer, hidden_num_layer1, node_num, feature_in, horizon, learning_rate, decay, batchsize, keep, early_stop_th, training_epochs, X_training, Y_training, X_val, Y_val, X_test, Y_test, scaler)


     end_time = datetime.datetime.now()

     print('Total training time: ', end_time-start_time)

     np.savetxt("../data/prediction_"+str(val_error)+"_"+str(test_error)+".csv", predic_res, delimiter = ',')
     np.savetxt("../data/prediction_Y.csv", test_Y, delimiter = ',')




