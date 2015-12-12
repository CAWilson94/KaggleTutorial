""" 
Going through Kaggle tutorial 
(uses interactive code boxes)

"""
import csv as csv
import numpy as np
import pandas as pd


titanic = pd.read_csv("train.csv")					# Read in training data
titanic_test = pd.read_csv("test.csv")				# Read in testing data

print (titanic.head(5)) 						# Print first 5 rows of csv
print (titanic.describe()) 						# Age only shows 714 rows, 
												# Others are 891. Need to 
print(titanic_test.head(5))						# clean data then Do same
print(titanic_test.describe())					# checks on the test data.

# Fill in empty cells with median of all 
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) 
# The age has to be the exact same values we filled the training set with.
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

print (titanic.describe()) 					# Data cleaning check
print (titanic_test.describe())				# Data cleaning check 

# Non numeric columns, how do we deal with them? We have to exclude our 
# non numeric columns when we are training or we could convert them to 
# numeric columns! These non numeric columns normally mean we are dealing
# with a classification problem. 
titanic.loc[titanic["Sex"] == "male","Sex"] = 0		# Convert male values to 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1 	# Convert female values to 1

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 		# Male values to 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1	# Female values to 1

# The Embarked column for both sets, had to be converte to numbers also. So, the 
# first step is to replace all missing values in the column. The most common value
# is S so lets assume everyone got on there.
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic_test["Embarked"] = titanic_test["Ambarked"].fillna("S")

# Fill in numeric values for training data
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# Fill in numeric values for testing data
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# Replace missing value in "Fare" column of test set
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())


"""
Logistic regression!! :D
------------------------
Using logistic regression to predict accuracy score when 
making predictions

The target, response or dependant variable is a simple, did they Survive
or not. So we are trying to assert the possibility of them surviving.
p = Survive (i.e. the event of them surviving occuring) 
p - 1 = the event not occuring. (Do not survive)

"""

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm
logReg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.
scores = cross_validation.cross_val_score(logReg, titanic[predictors], titanic["Survived"], cv=3) # cv=3, is the number of folds 	
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

# Processing the test set
# Need to take same steps with the test set as we have done with our training data
