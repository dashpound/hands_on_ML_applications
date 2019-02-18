###############################################################################
# Overview
###############################################################################

# Jump-Start for the Bank Marketing Study... Solution Code

# Case described in Marketing Data Science: Modeling Techniques
# for Predictive Analytics with R and Python (Miller 2015)

# jump-start code revised by Thomas W. Milller (2017/09/26)
# jump-start code modified to include one hot enconded features by John Kiley (2018/07/08)

# Scikit Learn documentation for this assignment:
# http://scikit-learn.org/stable/auto_examples/classification/
#   plot_classifier_comparison.html
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB.score
# http://scikit-learn.org/stable/modules/generated/
#   sklearn.linear_model.LogisticRegression.html
# http://scikit-learn.org/stable/modules/model_evaluation.html 
# http://scikit-learn.org/stable/modules/generated/
#  sklearn.model_selection.KFold.html

# prepare for Python version 3x features and functions
# comment out for Python 3.x execution
# from __future__ import division, print_function
# from future_builtins import ascii, filter, hex, map, oct, zip

###############################################################################
# Prep & EDA
###############################################################################

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# import base packages into the namespace for this program
import numpy as np
import pandas as pd

# use the full data set after development is complete with the smaller data set
# bank = pd.read_csv('bank-full.csv', sep = ';')  # start with smaller data set

# initial work with the smaller data set
bank = pd.read_csv('bank.csv', sep = ';')  # start with smaller data set
# examine the shape of original input data
print(bank.shape)

# drop observations with missing data, if any
bank.dropna()
# examine the shape of input data after dropping missing data
print(bank.shape)

# look at the list of column names, note that y is the response
list(bank.columns.values)

# look at the beginning of the DataFrame
bank.head()

###############################################################################
# Preprocess the data 
# create a function for onehot encoding columns
###############################################################################

# Import onehotencoder to preprocess this data
# import one hot encoder
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()

###############################################################################
# I want to look at demographic information
# select column to onehot encode
job_cat = bank['job'] 

# convert text names to a numerical value & create a vector with mapped names
job_cat_encoded, job_categories = job_cat.factorize()

# onehot encode job cat data 
job_cat_1hot = encoder.fit_transform(job_cat_encoded.reshape(-1,1))

# convert 1hot encode to array
job_cat_1hot = job_cat_1hot.toarray()

###############################################################################
# select column to onehot encode
marital_cat = bank['marital'] 

# convert text names to a numerical value & create a vector with mapped names
marital_cat_encoded, marital_categories = marital_cat.factorize()

# onehot encode job cat data 
marital_cat_1hot = encoder.fit_transform(marital_cat_encoded.reshape(-1,1))

# convert 1hot encode to array
marital_cat_1hot = marital_cat_1hot.toarray()

###############################################################################
# create unique vector for each job type
# automate this part, currently pulling value and assigning it
###############################################################################
# define variable
unemployed = job_cat_1hot[:,0]

# change data type to match the rest of the arrays (int64)
unemployed = unemployed.astype(np.int64)

###############################################################################
# define variable
services = job_cat_1hot[:,1]

# change data type to match the rest of the arrays (int64)
services = services.astype(np.int64)

###############################################################################
# define variable
management = job_cat_1hot[:,2]

# change data type to match the rest of the arrays (int64)
management = management.astype(np.int64)

###############################################################################
# define variable
blue_collar = job_cat_1hot[:,3]

# change data type to match the rest of the arrays (int64)
blue_collar = blue_collar.astype(np.int64)

###############################################################################
# define variable
self_employed = job_cat_1hot[:,4]

# change data type to match the rest of the arrays (int64)
self_employed = self_employed.astype(np.int64)

###############################################################################
# define variable
technician = job_cat_1hot[:,5]

# change data type to match the rest of the arrays (int64)
technician = technician.astype(np.int64)

###############################################################################
# define variable
entrep = job_cat_1hot[:,6]

# change data type to match the rest of the arrays (int64)
entrep = entrep.astype(np.int64)

###############################################################################
# define variable
admin = job_cat_1hot[:,7]

# change data type to match the rest of the arrays (int64)
admin = admin.astype(np.int64)

###############################################################################
# define variable
student = job_cat_1hot[:,8]

# change data type to match the rest of the arrays (int64)
student = student.astype(np.int64)

###############################################################################
# define variable
housemaid = job_cat_1hot[:,9]

# change data type to match the rest of the arrays (int64)
housemaid = housemaid.astype(np.int64)

###############################################################################
# define variable
retired = job_cat_1hot[:,10]

# change data type to match the rest of the arrays (int64)
retired = retired.astype(np.int64)

###############################################################################
# define variable
unknown = job_cat_1hot[:,11]

# change data type to match the rest of the arrays (int64)
unknown = unknown.astype(np.int64)

###############################################################################
# define variable
married = marital_cat_1hot[:,0]

# change data type to match the rest of the arrays (int64)
married = married.astype(np.int64)

###############################################################################
# define variable
single = marital_cat_1hot[:,1]

# change data type to match the rest of the arrays (int64)
single = single.astype(np.int64)

###############################################################################
# define variable
divorced = marital_cat_1hot[:,2]

# change data type to match the rest of the arrays (int64)
divorced = divorced.astype(np.int64)

###############################################################################
# Preprocessing from sample code
###############################################################################

# mapping function to convert text no/yes to integer 0/1
convert_to_binary = {'no' : 0, 'yes' : 1}

# define binary variable for having credit in default
default = bank['default'].map(convert_to_binary)

# define binary variable for having a mortgage or housing loan
housing = bank['housing'].map(convert_to_binary)

# define binary variable for having a personala loan
loan = bank['loan'].map(convert_to_binary)

# define response variable to use in the model
response = bank['response'].map(convert_to_binary)

###############################################################################
# Define Model dataset
###############################################################################

# gather three explanatory variables and response into a numpy array 
# here we use .T to obtain the transpose for the structure we want
model_data = np.array([np.array(default), np.array(housing), np.array(loan),
                       np.array(unemployed),np.array(services),
                       np.array(management),np.array(blue_collar),
                       np.array(self_employed), np.array(technician),
                       np.array(entrep), np.array(admin),
                       np.array(student), np.array(housemaid),
                       np.array(retired), np.array(unknown),
                       np.array(married),np.array(single),
                       np.array(divorced), np.array(response)]).T

model_data
# examine the shape of model_data, which we will use in subsequent modeling
print(model_data.shape)

###############################################################################
# Perform ML 
###############################################################################

# code adopted from jumpstart bank exercise 2

# cross-validation scoring code adapted from Scikit Learn documentation
from sklearn.metrics import roc_auc_score

# specify the set of classifiers being evaluated
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
names = ["Naive_Bayes", "Logistic_Regression"]
classifiers = [BernoulliNB(alpha=1.0, binarize=0.5, 
                           class_prior = [0.5, 0.5], fit_prior=False), 
               LogisticRegression()]

# dimensions of the additive model X input and y response
print('\nData dimensions:', model_data.shape)

# --------------------------------------------------------
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

# ten-fold cross-validation employed here
N_FOLDS = 10

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)
# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 
for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   note that 0:model_data.shape[1]-1 slices for explanatory variables
#   and model_data.shape[1]-1 is the index for the response variable    
    X_train = model_data[train_index, 0:model_data.shape[1]-1]
    X_test = model_data[test_index, 0:model_data.shape[1]-1]
    y_train = model_data[train_index, model_data.shape[1]-1]
    y_test = model_data[test_index, model_data.shape[1]-1]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, clf in zip(names, classifiers):
        print('\nClassifier evaluation for:', name)
        print('  Scikit Learn method:', clf)
        clf.fit(X_train, y_train)  # fit on the train set for this fold
        # evaluate on the test set for this fold
        y_test_predict = clf.predict_proba(X_test)
        fold_method_result = roc_auc_score(y_test, y_test_predict[:,1]) 
        print('Area under ROC curve:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
   
    index_for_fold += 1


cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod                 Area under ROC Curve', sep = '')     
print(cv_results_df.mean())   

# --------------------------------------------------------
# Select method and apply to specific test cases
# --------------------------------------------------------

# this problem expands exponentially (literally) for each feature added to the test
# the number of binary permutations is calculated 2^n; for this example, there are 262,144 combinations
# enhancement would be to use itertools to generate the test vector for each feature

my_default =       np.array([1, 1, 1, 1, 0, 0, 0, 0], np.int32)
my_housing =       np.array([1, 1, 0, 0, 1, 1, 0, 0], np.int32)
my_loan =          np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_unemploy =      np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_service =       np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_manage =        np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_blue_collar =   np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_self_employed = np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_tech =          np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_entrep =        np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_admin =         np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_student =       np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_housemaid =     np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_retired =       np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_unknown =       np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_married =       np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_single =        np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)
my_divorced =      np.array([1, 0, 1, 0, 1, 0, 1, 0], np.int32)

my_X_test = np.vstack([my_default, my_housing, my_loan, my_unemploy, my_service, 
                       my_manage, my_blue_collar, my_self_employed, my_tech, my_entrep, my_admin, my_student,
                       my_housemaid, my_retired, my_unknown, my_married,
                       my_single, my_divorced]).T

# fit logistic regression to full data set
clf = LogisticRegression()
X_train = model_data[:, 0:model_data.shape[1]-1]
y_train = model_data[:, model_data.shape[1]-1]
clf.fit(X_train, y_train)

# predict specific test cases
y_my_test_predict = clf.predict_proba(my_X_test)


# create DataFrame for displaying test cases and predicted probabilities
my_targeting_df = pd.DataFrame(np.hstack([my_X_test, y_my_test_predict]))
my_targeting_df.columns = ['default', 'housing', 'loan', 'unemploy', 'service', 'management',
                           'blue_collar', 'self_employed', 'tech', 'entrep', 'admin', 
                           'student', 'housemaid', 'retired', 'unknown', 'married',
                           'single', 'divorced', 'predict_NO', 'predict_YES']
print('\n\nLogistic regression model predictions for test cases:')
print(my_targeting_df) 
print('Caution using the table above; enhancement required to generate all permuations using itertools')
# for targeting, select group(s) with highest predictive probability 
# of responding to the promotional mailers