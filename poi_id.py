#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import numpy as np
import pprint

from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from tester import dump_classifier_and_data
import helper_functions


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Remove outliers

#helper_functions.scatterPlot(data_dict, ['poi','salary','bonus'], f1_name="salary", f2_name="bonus")
#helper_functions.scatterPlot(data_dict, ['poi','salary','bonus'], f1_name="feature 1", f2_name="feature 2")          
#helper_functions.checkNaN(data_dict)
#print data_dict.keys()  # check names in the data
#count_NaN = helper_functions.feature_NaN(data_dict)

del data_dict['TOTAL'], data_dict['THE TRAVEL AGENCY IN THE PARK'], data_dict['LOCKHART EUGENE E'] # remove outliers


### Task 2: select features

complete_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']  # complete feature list except for email address which is a string

best_features = helper_functions.get_k_best(data_dict, complete_features_list, 10)     
features_list = ['poi'] + best_features.keys()      
features_list.remove('total_stock_value')
features_list.remove('total_payments')
  

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
helper_functions.add_poi_feature(my_dataset, features_list)
#helper_functions.scatterPlot(data_dict, ['poi','fraction_from_poi','fraction_to_poi'], \
#            f1_name="fraction of emails from poi to this person", \
#            f2_name="fraction of emails from this person to poi") 


helper_functions.add_financial_aggregate(my_dataset, features_list)
#helper_functions.feature_NaN(data_dict)

#helper_functions.get_k_best(data_dict, features_list, 10) 

print features_list    

### scale features via min-max and update data_dict
#helper_functions.feature_scale(my_dataset, features_list)

### Extract scaled features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# scale features via min-max if not call helper_functions.feature_scale
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

"""Logistic Regression"""
from sklearn.linear_model import LogisticRegression

#################################################################################
# Optimize parameters by GridSearchCV
#max_exponent = 21
#parameters = {'C':[10** x for x in range(0, max_exponent, 3)], \
#                'tol': [10 ** (-x) for x in range(3, max_exponent, 3)],\
#                'class_weight': ['auto', None]}
#cv = StratifiedShuffleSplit(labels, n_iter=1000)                
#Logr = LogisticRegression()
#l_clf = GridSearchCV(Logr, parameters, scoring = helper_functions.scoring, cv = cv)
#l_clf.fit(features, labels)
#print l_clf.best_estimator_
#print l_clf.best_score_
#l_clf = l_clf.best_estimator_
#################################################################################

l_clf = LogisticRegression(C = 1, tol = 1e-15, class_weight = 'auto') # best estimator 


"""K-means Clustering"""

#################################################################################
#from sklearn.cluster import KMeans
#cv = StratifiedShuffleSplit(labels, n_iter=1000)
#parameters = {'n_clusters': [2], 'n_init' : [10, 20, 30], 'tol' : [10 ** (-x) for x in range(2, 6)]}
#k_clf = KMeans()
#k_clf = GridSearchCV(k_clf, parameters, scoring = helper_functions.scoring, cv = cv)
#k_clf.fit(features, labels)
#print k_clf.best_estimator_
#print k_clf.best_score_
#k_clf = k_clf.best_estimator_

#k_clf = KMeans(n_clusters=2, tol=0.001)
#################################################################################


"""Adaboost Classifier"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#################################################################################
# Optimize parameters by GridSearchCV
#cv = StratifiedShuffleSplit(labels, n_iter=1000)
#parameters = {'n_estimators' : [5, 10, 30, 40, 50, 100,150], 'learning_rate' : [0.1, 0.5, 1, 1.5, 2, 2.5], 'algorithm' : ('SAMME', 'SAMME.R')}
#ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8))
#adaclf = GridSearchCV(ada_clf, parameters, scoring = helper_functions.scoring, cv = cv)
#adaclf.fit(features, labels)
#
#print adaclf.best_estimator_
#print adaclf.best_score_
#
#a_clf = adaclf.best_estimator_
#################################################################################
a_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), algorithm='SAMME.R',learning_rate=2.5, n_estimators=40) # best estimator



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from sklearn.pipeline import Pipeline
clf = Pipeline([('scale', preprocessing.MinMaxScaler()), ('clf', a_clf)])
#clf = a_clf
dump_classifier_and_data(clf, my_dataset, features_list)
