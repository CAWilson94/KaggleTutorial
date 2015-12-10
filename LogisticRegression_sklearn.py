""" 
Going through Kaggle tutorial 
(uses interactive code boxes)

"""
import csv as csv
import numpy as np
import pandas as pd

titanic = pd.read_csv("train.csv")
print (titanic.head(5)) # Print first 5 rows of csv
print (titanic.describe()) # Age only shows 714 rows, others are 891. Need to clean data!

# Fill in empty cells with median of all 
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) 

print (titanic.describe()) #Check the data cleaning has worked :)

# Non numeric columns, how do we deal with them?
# We have to exclude our non numeric columns when we are training
# Or we could convert them to numeric columns! 

# Let's have a go of converting the gender(sex) column
# Select all male values in the column, replace them with 0
titanic.loc[titanic["Sex"] == "male","Sex"] = 0
# Select all female values in column, replace values with 1
titanic.loc[titanic["Sex"] == "female","Sex"] = 1

# Convert the embarked column 
# First step is to replace all missing values in the column
# Most common value is S so lets assume everyone got on there
titanic["Embarked"] = titanic["Embarked"].fillna("S")
# Fill in numeric values
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

"""
Logistic regression!! :D
------------------------
Using logistic regression to predict accuracy score when 
making predictions

"""

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm
logReg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.
scores = cross_validation.cross_val_score(logReg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())