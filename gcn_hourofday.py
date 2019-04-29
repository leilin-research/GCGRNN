

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
import json
from utils import normalize_adj, StandardScaler, masked_mae_tf




# Create model
def gcn(signal_in, weights_hidden, weights_other, biases_other, weights_A, biases, hidden_num, hidden_num1, node_num, horizon):
    
    print (signal_in.shape)
    signal_in_main = signal_in[:,:,:,0] #tf.concat([signal_in[:,:,:,0], signal_in[:,:,:,1]], axis = 2) #(?, 207, 24)
    #print (signal_in_main.shape)
    
    signal_in_main = tf.transpose(signal_in_main, [1, 0, 2]) # node_num, ?batch, feature_in
    feature_len = signal_in_main.shape[2] # feature vector length at the node of the input graph
    
    i = 0
    while i < hidden_num:

       # with open('distance_matrix.json') as f:
       #     dm = json.load(f)
	    
        #with open('duration_matrix.json') as f:
        #    tm = json.load(f)

        #dm = np.array(dm) # meters

        #dm = dm / 1000 # meters to km

        #tm = np.array(tm) # seconds to mins

        #tm = tm / 60
        #dm = np.exp(-dm)
        #tm = np.exp(-tm)

        #cr = pd.read_csv('corr_mx.csv', header = None)
        #cr = cr.values
        #dm = tf.convert_to_tensor(dm, dtype=np.float32)
        #tm = tf.convert_to_tensor(tm, dtype=np.float32)
        #cr = tf.convert_to_tensor(cr, dtype=np.float32)
        #Atem1 = normalize_adj(dm) #weights['alpha1'] * normalize_adj(dm)
        #Atem2 = normalize_adj(tm) #weights['alpha2'] * normalize_adj(tm) #normalize_adj(tm)
        #Atem3 = normalize_adj(cr)

        signal_in_main = tf.reshape(signal_in_main, [node_num, -1]) # node_num, batch*feature_in
        
        Adj = 0.5*(weights_A['A'+str(i)] + tf.transpose(weights_A['A'+str(i)])) #+ cr + dm + tm
        #Adj = cr + dm + tm
        Adj = normalize_adj(Adj)
        Z = tf.matmul(Adj, signal_in_main) # node_num, batch*feature_in 
        Z = tf.reshape(Z, [-1, int(feature_len)]) # node_num * batch, feature_in
        signal_output = tf.add(tf.matmul(Z, weights_hidden['h'+str(i)]), biases['b'+str(i)])
        signal_output = tf.nn.relu(signal_output) # node_num * batch, hidden_vec
        
        i += 1
        signal_in_main = signal_output # the sinal for next layer 
        feature_len = signal_in_main.shape[1] # feature vector length at hidden layers
        #print (feature_len)
    
    signal_in_other = signal_in[:,:,:,1] #tf.concat([signal_in[:,:,:,0], signal_in[:,:,:,1]], axis = 2) #(?, 207, 24)
    #print (signal_in_main.shape)
    print (signal_in_other.shape[2])
    signal_in_other = tf.reshape(signal_in_other, [-1, int(signal_in_other.shape[2])]) # node_num*batch, feature_in
    i = 0
    while i < hidden_num1:
        signal_output_ot = tf.add(tf.matmul(signal_in_other, weights_other['h'+str(i)]), biases_other['b'+str(i)])
        signal_output_ot = tf.nn.relu(signal_output_ot) # node_num * batch, hidden_vec
        
        i += 1
        signal_in_other = signal_output_ot # the sinal for next layer 
        #print (feature_len)

    
    #print (signal_output.shape)
    #print (signal_in[:, :, :, 1].shape)
    #signal_in_other = tf.reshape(signal_in_other, [-1, horizon])
    signal_output = tf.concat([signal_output, signal_output_ot], axis = 1)
    final_output = tf.add(tf.matmul(signal_output, weights_hidden['out']), biases['bout'])  # node_num * batch, horizon
    final_output = tf.reshape(final_output, [node_num, -1, horizon]) # node_num, batch, horizon
    final_output = tf.transpose(final_output, [1, 0, 2]) # batch, node_num, horizon
    final_output = tf.reshape(final_output, [-1, node_num*horizon]) # batch, node_num*horizon
 
    return final_output


def gcnn_ddgf(hidden_num_layer, node_num, feature_in, horizon, learning_rate, decay, batch_size, keep, early_stop_th, training_epochs, X_training, Y_training, X_val, Y_val, X_test, Y_test, scaler):
   
    n_output_vec = node_num * horizon # length of output vector at the final layer 
    
    early_stop_k = 0 # early stop patience
    display_step = 1 # frequency of printing results
    best_val = 10000
    traing_error = 0
    test_error = 0
    predic_res = []

    tf.reset_default_graph()

    batch_size = batch_size
    early_stop_th = early_stop_th
    training_epochs = training_epochs

    # tf Graph input and output
    X = tf.placeholder(tf.float32, [None, node_num, feature_in]) # X is the input signal
    Y = tf.placeholder(tf.float32, [None, n_output_vec]) # y is the regression output
    
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # define dictionaries to store layers weight & bias
    i = 0
    weights_hidden = {}
    weights_A = {}
    biases = {}
    vec_length = feature_in
    while i < len(hidden_num_layer):
        weights_hidden['h'+str(i)] = tf.Variable(tf.random_normal([vec_length, hidden_num_layer[i]]))
        biases['b'+str(i)] = tf.Variable(tf.random_normal([1, hidden_num_layer[i]]))
        weights_A['A'+str(i)] = tf.Variable(tf.random_normal([node_num, node_num]))
        vec_length = hidden_num_layer[i]
        i += 1
        
    
    weights_hidden['out'] = tf.Variable(tf.random_normal([hidden_num_layer[-1], horizon]))
    biases['bout'] = tf.Variable(tf.random_normal([1, horizon]))
 
    # Construct model
    hidden_num = len(hidden_num_layer) 
    hidden_num1 = len(hidden_num_layer1) 
    pred= gcn(X, weights_hidden, weights_A, biases, hidden_num, hidden_num1, node_num, horizon)
    pred = scaler.inverse_transform(pred)
    Y_true_tr = scaler.inverse_transform(Y)
    cost = tf.reduce_mean(tf.pow(pred - Y_true_tr, 2)) 

    pred_val= gcn(X, weights_hidden, weights_A, biases, hidden_num, hidden_num1, node_num, horizon)
    pred_val = scaler.inverse_transform(pred_val)
    Y_true_val = scaler.inverse_transform(Y)
    cost_val =  tf.reduce_mean(tf.pow(pred_val - Y_true_val, 2)) 

    pred_tes= gcn(X, weights_hidden, weights_A, biases, hidden_num, hidden_num1, node_num, horizon)
    pred_tes = scaler.inverse_transform(pred_tes)
    Y_true_tes = scaler.inverse_transform(Y)
    cost_tes = tf.reduce_mean(tf.pow(pred_tes - Y_true_tes, 2)) 
                                         
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):

            avg_cost = 0.
            num_train = X_training.shape[0]
            total_batch = int(num_train/batch_size)

            for i in range(total_batch):
                
                _, c = sess.run([optimizer, cost], feed_dict={X: X_training[i*batch_size:(i+1)*batch_size,], 
                                                      Y: Y_training[i*batch_size:(i+1)*batch_size,], 
                                                              keep_prob: keep})

                avg_cost += c * batch_size #/ total_batch 
                
            # rest part of training dataset
            if total_batch * batch_size != num_train:
                _, c, preds, trueval = sess.run([optimizer, cost, pred, Y_true_tr], feed_dict={X: X_training[total_batch*batch_size:num_train,], 
                                          Y: Y_training[total_batch*batch_size:num_train,],
                                                  keep_prob: keep})
                avg_cost += c * (num_train - total_batch*batch_size)
            
            avg_cost = np.sqrt(avg_cost / num_train)
            #Display logs per epoch step
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "Training RMSE=",                     "{:.9f}".format(avg_cost))
            # validation
            c_val = sess.run([cost_val], feed_dict={X: X_val, Y: Y_val,  keep_prob:1})
            c_val = np.sqrt(c_val[0])
            print("Validation RMSE: ", c_val)
            # testing
            c_tes, preds, Y_true = sess.run([cost_tes, pred_tes, Y_true_tes], feed_dict={X: X_test,Y: Y_test, keep_prob: 1})
            c_tes = np.sqrt(c_tes)

            if c_val < best_val:
                best_val = c_val
                # save model
                #saver.save(sess, './bikesharing_gcnn_ddgf')
                test_error = c_tes
                traing_error = avg_cost
                predic_res = preds
                early_stop_k = 0 # reset to 0

            # update early stopping patience
            if c_val >= best_val:
                early_stop_k += 1

            # threshold
            if early_stop_k == early_stop_th:
                break
            

        print("epoch is ", epoch)
        print("training RMSE is ", traing_error)
        print("Optimization Finished! the lowest validation RMSE is ", best_val)
        print("The test RMSE is ", test_error)
    
    #test_Y = Y_test
    #test_error = np.sqrt(test_error)
    return best_val, predic_res,Y_true,test_error

def gcnn_ddgf_mae(hidden_num_layer, hidden_num_layer1, node_num, feature_in, horizon, learning_rate, decay, batch_size, keep, early_stop_th, training_epochs, X_training, Y_training, X_val, Y_val, X_test, Y_test, scaler, channel = 2):
   
    n_output_vec = node_num * horizon # length of output vector at the final layer 
    
    early_stop_k = 0 # early stop patience
    display_step = 1 # frequency of printing results
    best_val = 10000
    traing_error = 0
    test_error = 0
    predic_res = []

    tf.reset_default_graph()

    batch_size = batch_size
    early_stop_th = early_stop_th
    training_epochs = training_epochs

    # tf Graph input and output
    X = tf.placeholder(tf.float32, [None, node_num, feature_in, channel]) # X is the input signal
    Y = tf.placeholder(tf.float32, [None, n_output_vec]) # y is the regression output
    
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # define dictionaries to store layers weight & bias
    i = 0
    weights_hidden = {}
    weights_other = {}
    weights_A = {}
    biases = {}
    biases_other = {}
    vec_length = feature_in
    while i < len(hidden_num_layer):
        weights_hidden['h'+str(i)] = tf.Variable(tf.random_normal([vec_length, hidden_num_layer[i]]))
        biases['b'+str(i)] = tf.Variable(tf.random_normal([1, hidden_num_layer[i]]))
        weights_A['A'+str(i)] = tf.Variable(tf.random_normal([node_num, node_num]))
        vec_length = hidden_num_layer[i]
        i += 1
    i = 0
    vec_length = feature_in
    while i < len(hidden_num_layer1):
        weights_other['h'+str(i)] = tf.Variable(tf.random_normal([vec_length, hidden_num_layer1[i]]))
        biases_other['b'+str(i)] = tf.Variable(tf.random_normal([1, hidden_num_layer1[i]]))
        vec_length = hidden_num_layer1[i]
        i += 1
    weights_hidden['out'] = tf.Variable(tf.random_normal([hidden_num_layer[-1] + hidden_num_layer1[-1], horizon]))
    biases['bout'] = tf.Variable(tf.random_normal([1, horizon]))
 
    # Construct model
    hidden_num = len(hidden_num_layer) 
    hidden_num1 = len(hidden_num_layer1)
    pred= gcn(X, weights_hidden, weights_other, biases_other, weights_A, biases, hidden_num, hidden_num1,node_num, horizon)
    pred = scaler.inverse_transform(pred)
    Y_true_tr = scaler.inverse_transform(Y)
    cost = masked_mae_tf(pred, Y_true_tr, 0)
    # regularization
    #l1_regularizer = tf.contrib.layers.l1_regularizer(
    #scale=0.00005, scope=None
    #)
    #weights = tf.trainable_variables() # all vars of your graph
    #regularization_penalty = tf.reduce_sum([ tf.nn.l2_loss(v) for v in weights ])
    #print(weights)
    #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    #cost = cost + 0.0005*regularization_penalty
    #i = 0
    #while i < len(reg_weight):
    #    cost += reg_weight[i]*tf.reduce_sum(tf.abs(weights_A['A'+str(i)]))
    #    i += 1

    pred_val= gcn(X, weights_hidden, weights_other, biases_other, weights_A, biases, hidden_num, hidden_num1,node_num, horizon)
    pred_val = scaler.inverse_transform(pred_val)
    Y_true_val = scaler.inverse_transform(Y)
    cost_val = masked_mae_tf(pred_val, Y_true_val, 0) 

    pred_tes= gcn(X, weights_hidden, weights_other, biases_other, weights_A, biases, hidden_num, hidden_num1,node_num, horizon)
    pred_tes = scaler.inverse_transform(pred_tes)
    Y_true_tes = scaler.inverse_transform(Y)
    cost_tes = masked_mae_tf(pred_tes, Y_true_tes, 0) 
                                         
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):

            avg_cost = 0.
            num_train = X_training.shape[0]
            total_batch = int(num_train/batch_size)

            for i in range(total_batch):
                
                _, c = sess.run([optimizer, cost], feed_dict={X: X_training[i*batch_size:(i+1)*batch_size,], 
                                                      Y: Y_training[i*batch_size:(i+1)*batch_size,], 
                                                              keep_prob: keep})

                avg_cost += c * batch_size #/ total_batch 
                
            # rest part of training dataset
            if total_batch * batch_size != num_train:
                _, c, preds, trueval = sess.run([optimizer, cost, pred, Y_true_tr], feed_dict={X: X_training[total_batch*batch_size:num_train,], 
                                          Y: Y_training[total_batch*batch_size:num_train,],
                                                  keep_prob: keep})
                avg_cost += c * (num_train - total_batch*batch_size)
            
            avg_cost = avg_cost / num_train
            #Display logs per epoch step
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "Training MAE=",                     "{:.9f}".format(avg_cost))
            # validation
            c_val = sess.run([cost_val], feed_dict={X: X_val, Y: Y_val,  keep_prob:1})
            c_val = c_val[0]
            print("Validation MAE: ", c_val)
            # testing
            c_tes, preds, Y_true = sess.run([cost_tes, pred_tes, Y_true_tes], feed_dict={X: X_test,Y: Y_test, keep_prob: 1})
            c_tes = c_tes

            if c_val < best_val:
                best_val = c_val
                # save model
                # saver.save(sess, 'traffic_speed_gcnn_ddgf')
                test_error = c_tes
                traing_error = avg_cost
                predic_res = preds
                early_stop_k = 0 # reset to 0

            # update early stopping patience
            if c_val >= best_val:
                early_stop_k += 1

            # threshold
            if early_stop_k == early_stop_th:
                break
            

        print("epoch is ", epoch)
        print("training MAE is ", traing_error)
        print("Optimization Finished! the lowest validation MAE is ", best_val)
        print("The test MAE is ", test_error)
    
    #test_Y = Y_test
    #test_error = np.sqrt(test_error)
    return best_val, predic_res,Y_true,test_error



