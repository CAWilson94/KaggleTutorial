"""
To improve the accuracy of my submission, 
using the dataquest titanic tutorial
"mission 2"

Author @Charlotte Alexandra Wilson
Data   December 2015

"""
import csv as csv
import numpy as np
import pandas as pd

titanic = pd.read_csv("train.csv")				# Read in training data

# Since our submission wasn't very high scoring, there are 3 ways
# in which we can improve it:
# (1) Use a better machine learning algorithm
# (2) Generate better features
# (3) Combine multiple machine learning algorithms (this is interesting...)

# First lets try out Random Forests in place of LogisticRegression
# Random forests are multiple Decision trees, which we take an average of 
# for our prediction. This is to avoid overfitting when using a single 
# Decision tree 

# Import cross validation as before
from sklearn import cross_validation
# Import random forests estimator/model
from sklearn.ensemble import RandomForestClassifier 

# Our feature/predictor columns
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
 
# Instantiate the model with the default paramaters 
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split 
# min_samples_leaf is the minimum number of samples we have at the place 
# where a tree branch ends (minimum number of leaves of child nodes)

randFor = RandomForestClassifier(random_state=1, n_estimators=10,
min_samples_split=2,min_samples_leaf=1)
