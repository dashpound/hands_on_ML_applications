# =============================================================================
# Documentation
# =============================================================================
# Code created for homework assigment 5 for MSDS 422.
# Assignment is to train various classification ML models on the MNIST dataset 

# Code modified by John Kiley - 07/28/2018

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html 
#http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py 
#http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html 

# =============================================================================
# Set up working environment
# =============================================================================

# import base packages into the namespace for this program
import pandas as pd
import numpy as np
import time

RANDOM_STATE=42
# =============================================================================
# Retrieve data
# =============================================================================

# =============================================================================
# mldata.org is currently down; if it were not, the data would be pulled with this code
# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
# =============================================================================

# Read in files
mnist_X = pd.read_csv('mnist_X.csv')
mnist_y = pd.read_csv('mnist_y.csv')

# Check number of observations per set
print('\nShape of full dataset:', mnist_X.shape)
print('Shape of vector of classifiers:', mnist_y.shape)
print('\n---------------------------------------------------------------------')

# =============================================================================
# Principal Componets Analysis
# =============================================================================
# Import packages
from sklearn.decomposition import PCA

# Time and loop
replications = 10  # repeat trial the ten times
x = [] # empty list for storing test results
n = 0  # initialize count
while (n < replications): 
    start_time = time.clock()
    
    # Define PCA model
    pca = PCA(n_components=0.95, svd_solver='full', random_state=RANDOM_STATE)

       #fit PCA    
    pca.fit(mnist_X)
    #output PCA
    X_reduced = pca.fit_transform(mnist_X)
    
    # End the timer
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    x.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", x[n], "milliseconds\n") 
    n = n + 1
    runtime = end_time - start_time  # seconds of wall-clock time
    print('\n---------------------------------------------------------------------')

print('\nThe average time for',replications,'trials is',round(np.mean(x),0),'milliseconds')

# =============================================================================
# Split the data into training and testing
# =============================================================================

X_train = X_reduced[:60000]
X_test = X_reduced[60000:]
y_train = mnist_y[:60000]
y_test = mnist_y[60000:]

# View shape of train test split
print('\nShape of model data:',
      '\nData Set: (Observations, Variables)')
print('X_train:', X_train.shape)
print('X_test:',X_test.shape)
print('y_train:', y_train.shape)
print('y_test:',y_test.shape)
print('\n---------------------------------------------------------------------')

# =============================================================================
# Select models and hyper parameters
# =============================================================================
# Import packages 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Record the amount of time to fit the model and evaluate the performance
replications = 10  # repeat the trial 10 times
x = [] # empty list for storing test results
n = 0  # initialize count
while (n < replications): 
    start_time = time.clock()
    # Set list of names and hyper parameters

    names = ["Random Forest, with PCA"]
    models = [RandomForestClassifier(max_features='sqrt', bootstrap=True,
                                 n_estimators=10, random_state=RANDOM_STATE)]

    # Select model from list of models
    name = names[0]
    clf = models[0]

    # Flatten/ravel vector of answers
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Train the model
    print('\nClassifier training leveraging a', name, ' model')
    print('  Scikit Learn method:', clf)

    # Fit on the train set
    clf.fit(X_train, y_train)  

    # Predict values
    y_test_predict = clf.predict(X_test)
    print('\n---------------------------------------------------------------------')

# =============================================================================
# Test the effectiveness of the model
# =============================================================================
    # Evaluate the model
    print('\nClassifier training, evaluation of', name, ' model')
    print('\n',classification_report(y_test, y_test_predict))


    # End the timer
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time
    x.append(runtime * 1000)  # report in milliseconds 
    print("replication", n + 1, ":", x[n], "milliseconds\n") 
    n = n + 1
    runtime = end_time - start_time  # seconds of wall-clock time
    print('\n---------------------------------------------------------------------')

print('\nThe average time for',replications,'trials is',round(np.mean(x),0),'milliseconds')