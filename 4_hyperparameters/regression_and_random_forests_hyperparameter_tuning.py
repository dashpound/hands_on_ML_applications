# =============================================================================
# Documentation
# =============================================================================

# Boston Housing Study (Python)
# using data from the Boston Housing Study case
# as described in "Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python" (Miller 2015)

# Here we use data from the Boston Housing Study to evaluate
# regression modeling methods within a cross-validation design.

# program revised by Thomas W. Milller (2017/09/29)
# program revised by John Kiley (2018/07/14)
# program revised by John Kiley (2018/07/21)

# Scikit Learn documentation for this assignment:
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.model_selection.KFold.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LinearRegression.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Ridge.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.Lasso.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.ElasticNet.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.metrics.r2_score.html

# Textbook reference materials:
# Geron, A. 2017. Hands-On Machine Learning with Scikit-Learn
# and TensorFlow. Sebastopal, Calif.: O'Reilly. Chapter 3 Training Models
# has sections covering linear regression, polynomial regression,
# and regularized linear models. Sample code from the book is 
# available on GitHub at https://github.com/ageron/handson-ml

# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

# =============================================================================
# Prepare the data
# =============================================================================

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import numpy as np
import pandas as pd

# read data for the Boston Housing Study
# creating data frame restdata
boston_input = pd.read_csv('boston.csv')

# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([boston.mv,\
    boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

print(prelim_model_data)
# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)

# =============================================================================
# exploratory data analysis 
# =============================================================================

# import visualization packages
import matplotlib.pyplot as plt

# create a list of plots to make
cols = list(boston)
print(cols)

# loop through columns to produce histograms of the various metrics
for i in cols: 
     fig, axes = plt.subplots()
     x = axes.hist(x=boston[i], alpha=0.75)
     axes.set_xlabel('Frequency')
     axes.set_ylabel(i, labelpad=20)
     axes.set_title('Histogram of ' + i, 
             fontsize=13)
     fig.savefig('boston_details_'+ i + '.pdf', 
                     bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                     orientation='portrait', papertype=None, format=None, 
                     transparent=True, pad_inches=0.25, frameon=None)  
     plt.show()     

     fig, axes = plt.subplots()
     x = axes.scatter(x= boston['mv'], y=boston[i], alpha=0.25)
     axes.set_xlabel('Median Home Value in 1970$s')
     axes.set_ylabel(i, labelpad=20)
     axes.set_title('Scatter MV vs ' + i, 
             fontsize=13)
     fig.savefig('boston_scatter'+ i + '.pdf', 
                     bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                     orientation='portrait', papertype=None, format=None, 
                     transparent=True, pad_inches=0.25, frameon=None)  
     plt.show() 

# =============================================================================
# select models |||| linear models, Random Forests, and Extra Trees
# =============================================================================

# modeling routines from Scikit Learn packages
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from math import sqrt  # for root mean-squared error calculation

# Set list of names and hyper parameters
names = ["Linear Regression", "Ridge Regression",
         "Lasso Regression", "Elastic Net", "Random Forest",
         "Extra Trees"]
models = [LinearRegression(), Ridge(alpha=1, solver='cholesky'), 
          Lasso(alpha=0.1), ElasticNet(alpha=0.1, l1_ratio=0.5), 
          RandomForestRegressor(n_estimators=500, max_features='log2', bootstrap=True),
          ExtraTreesRegressor()]

# =============================================================================
# kfolds validation |||| five folds
# =============================================================================

# --------------------------------------------------------
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# ten-fold cross-validation employed here
N_FOLDS = 5

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))
cv_results2 = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   note that 1:model_data.shape[1] slices for explanatory variables
#   and model_data.shape[0] is the index for the response variable    
    X_train = model_data[train_index, 0:model_data.shape[1]-1]
    X_test = model_data[test_index,  0:model_data.shape[1]-1]
    y_train = model_data[train_index, model_data.shape[1]-1]
    y_test = model_data[test_index, model_data.shape[1]-1]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(names, models):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  # fit on the train set for this fold
        # evaluate on the test set for this fold
        y_test_predict = clf.predict(X_test)
        fold_method_result = sqrt(mean_squared_error(y_test,y_test_predict))
        fold_method_result2 = r2_score(y_test,y_test_predict)
        print('MSE:', fold_method_result)
        print('R2 Score:', fold_method_result2)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        cv_results2[index_for_fold, index_for_method] = fold_method_result2
        index_for_method += 1
  
    index_for_fold += 1

# set up the R2 and MSE score outputs
cv_results_df = pd.DataFrame(cv_results)
cv_results_df2 = pd.DataFrame(cv_results2)
cv_results_df.columns = names
cv_results_df2.columns = names

cv_results_df2

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 MSE', sep = '')     
print(cv_results_df.mean())  
print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 R2', sep = '')     
print(cv_results_df2.mean())  
print('\n----------------------------------------------')

# =============================================================================
# Select method, train a model, and create predicted regression line
# =============================================================================

# fit regression to full data set
clf = models[4]
X_train = model_data[:, 0:model_data.shape[1]-1]
y_train = model_data[:, model_data.shape[1]-1]
clf.fit(X_train, y_train)

# view the shape of the data 
X_train.shape
y_train.shape

# predict
y_predict = clf.predict(X_train)

# =============================================================================
# View relevant statistics for the specific model
# =============================================================================

# Store feature importance as a data frame
FI = pd.DataFrame({'feature': cols[:12],
                   'feature_importance':clf.feature_importances_})

# sort dataframe by feature importance
FI.sort_values(by=['feature_importance'], ascending=False, inplace=True)

# print feature importances
print('Feature Importance:')
print(FI)
print('\n----------------------------------------------')    

# =============================================================================
# View Results
# =============================================================================
from matplotlib.lines import Line2D

fig, axes = plt.subplots()
axes.scatter(y_train,y_predict, color='blue', alpha=0.25)
axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c="red")
axes.set_xlabel('actual values')
axes.set_xlim(-4,4)
axes.set_ylabel('predicted values', labelpad=20)
axes.set_ylim(-4,4)
axes.set_title(names[4], 
             fontsize=13)

legend_elements = [Line2D([0], [0], color='R', lw=1, label='Perfect Prediction'),
                   Line2D([0], [0], marker='o', color='w', label='Actual v Predicted',
                          markerfacecolor='b', markersize=15),]
                   
axes.legend(handles=legend_elements)
fig.savefig('regression_output'+ '.pdf', 
                     bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
                     orientation='portrait', papertype=None, format=None, 
                     transparent=True, pad_inches=0.25, frameon=None)  