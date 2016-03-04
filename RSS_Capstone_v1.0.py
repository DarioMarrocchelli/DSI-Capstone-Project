import csv as csv 
import numpy as np
import pandas as pd
import pylab as py
import matplotlib.pyplot as plt
import math
import datetime
from time import time
from Functions import RMSEP


# This is a python script I have written for a Kaggle Competition (https://www.kaggle.com/c/rossmann-store-sales)
# The structure of this script is as follows:
# 1) Read the data and perform some basic data cleaning
# 2) Engineer a series of features to be used in a ML algorithm
# 3) Train a ML algorithm (reandom forest regressor or similar) and test it
# 4) Generate a csv file for a Kaggle submission

##
## Part 1: reading and cleaning data
##

# Time variable used to estimate the time it takes to run this script
t0 = time()

# Open up the csv files into Pandas dataframes
# Opens the train.csv and store.csv and merges them together into data

train = pd.DataFrame.from_csv(open('/Users/dadda/Dropbox (MIT)/Kaggle Competitions/RSS/train.csv', 'rb',), index_col=None) 
# Removes all thos rows that have zero sales; this is because zero sales are not scored in the evaluation
train = train[train.Sales > 1]
extra_data = pd.DataFrame.from_csv(open('/Users/dadda/Dropbox (MIT)/Kaggle Competitions/RSS/store.csv', 'rb',), index_col=None) 
data = pd.merge(train, extra_data, on = 'Store')

# Only use a fraction of the dat; trick to speed up the calculation; generally not used
#data = data[0:len(data)/5]


# # Opens the test.csv and store.csv and merges them together into data
temp = pd.DataFrame.from_csv('/Users/dadda/Dropbox (MIT)/Kaggle Competitions/RSS/test.csv', index_col=None)
test = pd.merge(temp, extra_data, on = 'Store')


# Data cleaning, removing NaN from the files
# Note that I am not tackling all the variables with NaN here, because there are many I won't use
test.loc[(test.Open.isnull()), 'Open'] = 1
data.loc[(data.CompetitionDistance.isnull()), 'CompetitionDistance'] = data['CompetitionDistance'].mean()
test.loc[(test.CompetitionDistance.isnull()), 'CompetitionDistance'] = data['CompetitionDistance'].mean()

##
## Part 2: feature engineering
##

# Many features are commented out because in the end I decided these were not necessary 
# (see also iPython notebbok on GitHub or writeup on Medium.com)

data['Year'] = pd.to_datetime(data['Date']).dt.year
#data['Month'] = pd.to_datetime(data['Date']).dt.month
#data['Day'] = pd.to_datetime(data['Date']).dt.day
#data['Date2'] = pd.to_datetime(data['Date']).dt.month
#data['StateHoliday2'] = data['StateHoliday'].map( {'a': 1, 'b': 2, 'c': 3, '0': 0, 0:0} ).astype(int)
#data['StoreType2'] = data['StoreType'].map( {'a': 1, 'b': 2, 'c': 3, 'd': 4} ).astype(int)
#data['Assortment2'] = data['Assortment'].map( {'a': 1, 'b': 2, 'c': 3, 'd': 4} ).astype(int)
#data['PromoInterval'] = data['PromoInterval'].map( {'Feb,May,Aug,Nov': 1, 'Jan,Apr,Jul,Oct': 2, 'Mar,Jun,Sept,Dec': 3, 0:0} ).astype(int)
#data['LaborDay'] = np.where( ((data['Month']) == 5) & ((data['Day']) == 1), 1, 0)
#data['Competitors'] = np.where( (data['DayOfWeek']) == 7, ((data['CompetitionDistance'])/6000), -1)
#data['Competitors'] = data['Competitors'].astype(int)
#dictionary = { 0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, -1:2}
#data['Competitors'] = data['Competitors'].map( dictionary )


test['Year'] = pd.to_datetime(test['Date']).dt.year
#test['Month'] = pd.to_datetime(test['Date']).dt.month
#test['Day'] = pd.to_datetime(test['Date']).dt.day
#test['Date2'] = pd.to_datetime(test['Date']).dt.month#.dtype='datetime64[ns]
#test['StateHoliday2'] = test['StateHoliday'].map( {'a': 1, 'b': 2, 'c': 3, '0': 0, 0:0 } ).astype(int)
#test['StoreType2'] = test['StoreType'].map( {'a': 1, 'b': 2, 'c': 3, 'd': 4} ).astype(int)
#test['Assortment2'] = test['Assortment'].map( {'a': 1, 'b': 2, 'c': 3, 'd': 4} ).astype(int)
#test['PromoInterval2'] = test['PromoInterval'].map( {'Feb,May,Aug,Nov': 1, 'Jan,Apr,Jul,Oct': 2, 'Mar,Jun,Sept,Dec': 3, 0:0} ).astype(int)
#test['Promo'] = 10000*test['Promo']
#test['LaborDay'] = np.where( ((test['Month']) == 5) & ((test['Day']) == 1), 1, 0)
#test['Competitors'] = np.where( (test['DayOfWeek']) == 7, ((test['CompetitionDistance'])/6000), -1)
#test['Competitors'] = test['Competitors'].astype(int)
#test['Competitors'] = test['Competitors'].map( dictionary )


# Using log(sales) rather than sales. This improves the overall model precision
# by weighing outliers less

data['Sales'] = np.log(data['Sales']+1)

average = np.exp(data['Sales'].mean())
print('The average sale per store is :' + str(average))

# Creates the label array
labels_train = data['Sales'].values

# Names of features (that were originally in the store, train or test files) we want to drop
features_dropped_store = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Promo2']
features_dropped_train = ['StateHoliday', 'Open', 'SchoolHoliday', 'Date', 'Customers', 'Sales']
features_dropped_test =  ['StateHoliday', 'Open', 'SchoolHoliday', 'Date', 'Id']

# Drops all the features in data we do not want to use
data = data.drop(features_dropped_train, axis = 1) 
data = data.drop(features_dropped_store, axis = 1)

# Saves the ID column into a series; I will need this when creating the submission file
PassID = test['Id']

# Drops all the features in test we do not want to use
test = test.drop(features_dropped_test, axis=1) 
test = test.drop(features_dropped_store, axis = 1)

print('We are using the following features: ' + str(data.columns.values))

# Creates the features array
features_train = data.values
features_test = test.values


#
# PART 3: MACHINE LEARNING!
#

# Imports a bunch of ML regressors and tools

from sklearn.metrics import r2_score
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import grid_search, svm
from sklearn.tree import DecisionTreeRegressor

# Several ML regressors; RF seem to be the best

clf = RandomForestRegressor(n_estimators = 10, n_jobs = -1, max_depth = 1119)
#clf = SVR(C=10)
#clf = DecisionTreeRegressor(min_samples_split = 8)
#clf = LinearRegression()
#clf = AdaBoostRegressor(n_estimators = 100)
#clf = GradientBoostingRegressor(n_estimators = 100)
#clf = BaggingRegressor()

# Parameters for grid_search; 
# RF needs very little parameter tweaking, so I commented these out

#parameters = {'max_features':['auto', 'sqrt', 'log2']}
#svr = RandomForestRegressor(n_jobs = 1, n_estimators = 1)
#clf = grid_search.GridSearchCV(svr, parameters)
#print clf.best_estimator_

# N-fold cross-validation
from sklearn import cross_validation

ss = cross_validation.ShuffleSplit(len(labels_train), n_iter=5, test_size=0.3, random_state=0)
scores = cross_validation.cross_val_score(clf, features_train, labels_train, cv=ss, scoring = 'r2')
print("Average accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

# Fits the regressor, uses the regressor for predicting sales and then calculates various metrics
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_train)
print('Trainingtime is: ' + str(time() - t0))


# This is the same metric as used on the Kaggle leaderboard
print('The RSMEP of the model is: ' + str(round(RMSEP(pred, labels_train),2)))


#
# PART 4: Generates submission file for Kaggle leaderboard
#
pred = clf.predict(features_test)
pred = np.exp(pred) - 1
print('Average sale during the test period: ' + str(pred.mean()))

solution = pd.DataFrame(dict(pred = pred, PassID = PassID))
solution.columns = ['Id', 'Sales']
solution = solution.sort('Id')
solution.to_csv('/Users/dadda/Dropbox (MIT)/Kaggle Competitions/RSS/solution.csv', index = False)


#
# EXTRA BITS OF DISCARDED CODE
#

# The lines below can be used to select the best features; 
# however, I prefer to rely on my exploratory analysis, as I fear over-fitting...

#from sklearn.feature_selection import SelectKBest
#selector = SelectKBest(k=3)
#selector.fit(features_train, labels_train)
#print(selector.scores_)

# I have made a function that uses a OneHotEncoder to create dummy variables
# for those features that have multiple categorical values
# As I explain in the write-up, this is useless for decision tree based regressors

#def OHE(features):
    
#    from sklearn.preprocessing import OneHotEncoder
#    enc = OneHotEncoder()
#    features = enc.fit_transform(features)
#    return features

#features_train = OHE(features_train)
#features_test  = OHE(features_test)

#features_test = enc.transform(features_test)

#simple_average = [average] * 844338
#print('The accuracy of simply using the average is: ' + str(round(100*(r2_score(labels_train, simple_average)),2)))