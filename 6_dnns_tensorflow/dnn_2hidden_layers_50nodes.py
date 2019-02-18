# =============================================================================
# TensorFlow/TensorBoard Neural Network MNIST Data
# Model with 2 hidden layers and 50 Nodes per layer
# 
# Code adapted by John Kiley 08/03/2018
#
# Conde adatpted from Thomas W. Miller 2018-01-07 
# tensorflow-to-tensorboard-linear-regression.py
# Code adapted from Geron Hands on ML - Chapter 10
#
# =============================================================================
#
# TensorFlow documentation 
# https://www.tensorflow.org/api_docs/python
# https://www.tensorflow.org/tutorials/index.html
#
# Reviewing TensorFlow terminology:
# tensor = multidimensional data array... vector, matrix, tensor
# graph = structure of the model, representing the model/neural network
# operation = mathematical operation on the graph
# session = environment for running operations on the graph
# placeholder = observed data object (fixed values from the input data)
# Variable = used for weights and biases to be fit in learning process
# learning_rate = used in optimization operations
#%%
# =============================================================================
# Setup work environment
# =============================================================================

# ensure common functions across Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time

start_time = time.clock()

RANDOM_SEED = 9999  # to ensure reproducibility of study results
#%%
# =============================================================================
# Import data/Generate data
# =============================================================================

# Specify general machine learning meta-parameters
N_EPOCHS = 100
LEARNING_RATE = 0.01   # meta-parameter in machine learning
BATCH_SIZE =50

print('\n-----------------------------------')
print('Import Data for Neural Network')
print('-----------------------------------')

# standard normal random number generator used to generate data and noise
np.random.seed(RANDOM_SEED)

# Pull data from 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# three way partitioning of data: train, validate, and test
#%%
# =============================================================================
# Setup Logs for Tensorflow
# =============================================================================
print('\n-----------------------------------')
print('    Set Up Logs for TensorBoard')
print('-----------------------------------')
from datetime import datetime  # use for time-stamps in activity log

# Define location and format for TensorFlow log a la Geron (2017) page 242.
# By settng up these directories we are preparing to visualize graphs
# and to tract the progress of learning in TensorBoard on a web browser.
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf-logs'
logdir = 'tmp/dnn_mnist_study3-{}/'.format(root_logdir, now)

#%%
# =============================================================================
# Construct Graph in Tensor flow
#=============================================================================
print('\n-----------------------------------')
print('TensorFlow Graph Construction Phase')
print('-----------------------------------')

n_inputs = 28*28  # MNIST
n_hidden1 = 50
n_hidden2 = 50
n_outputs = 10

# Name graph
g3 = tf.Graph()
 
# Construct the graph
with g3.as_default():

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    def neuron_layer(X, n_neurons, name, activation=None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            Z = tf.matmul(X, W) + b
            if activation is not None:
                return activation(Z)
            else:
                return Z
    
    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                               activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                               activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name="outputs")
    
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    
    learning_rate = 0.01
    
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
# could examine the graph operations for the linear regression mode
# but this would be a long listing... better to refer to a visualization    
# print('\nOperations on graph g:\n')
# for operation in g.get_operations():
#     print(operation)  
      
# send graph information to TensorFlow log for display in TensorBoard
file_writer = tf.summary.FileWriter(logdir, graph = g3) 

#%%
# =============================================================================
# Start a tensor session and execute a graph
# =============================================================================

print('\n-----------------------------------')
print('  TensorFlow Graph Execution Phase')
print('-----------------------------------') 
# Operations in TensorFlow do not do anything until we run them in a session.
# The constant is not actually set to a value in TensorFlow 
# until we set up and run a TensorFlow session

n_epochs = N_EPOCHS
batch_size = BATCH_SIZE

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session(graph = g3) as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "./my_model_final_study3.ckpt")
    sess.close()  # here all resources should be cleared

# Close the log file at end of program
# The file will be used as input to the TensorBoard application.
file_writer.close()

#%%
# =============================================================================
# Instructions for opening the tensorboard
# =============================================================================

# While working in the Terminal application on a Mac, for example,
# we locate ourselves in the working directory for the current
# TensorFlow program, and we enter commands to start the
# TensorBoard web browser application:
# 
#             tensorboard --logdir tmp/regression-demo-tf-logs

# TensorBoard is an application that runs within
# a modern browser, viewable by typing
#
#             localhost:6006

# TensorBoard reads TensorFlow events files, which contain summary 
# data that we generate while running TensorFlow sessions.
# TensorBoard also creates visualizations of TensorFlow graphs.

print('\n------------------------------------')
print('TensorBoard Visualizations Available')
print('------------------------------------') 
print('\nIn Mac OSX Terminal or Linux command shell, ')
print('change directory to current directory of this program and type:',
      '\n\n   tensorboard --logdir tmp/dnn_mnist-tf-logs')
print('\nThen in URL address field of modern browser, type:',
      '\n\n    localhost:6006')

print('\n------------------------------------')
print('End-to-end Runtime')
print('------------------------------------') 
end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time
print('\n---------------------------------------------------------------------')

print('\nThe average time end-to-end is',runtime,'seconds')