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
On to actual Machine Learning!

Cross Validation:
-----------------
We can now use linear regression  to make predictions on our training set!
We want to train algorithm on different data than we make predictions on.

This is critical if we want to avoid overfitting - when a model fits itself
to noise, not signal.

Cross validation is a good way to avoid overfitting
To cross validate you split your data into some number of parts, lets
use 3 as an example:

Combine 1st two parts, train a model, make predictions on the 3rd.
Combine 1st and 3rd parts, train a model, make predictions on the 2nd.
Combine 2nd and 3rd parts, train a model, make predictions on the 1st.

This way, we make predictions on the whole dataset without ever evaluating
accuracy on the same dataset we train our model using.

Making Predictions:
------------------- 
Use the Sci-kit learn library to make predictions
Use a helper from Sci-kit learn to split our data into crossvalidation folds
Then train an algorithm for each fold, and make predictions,
at the end we have a list of predictions, each list item containing 
predictions for the corresponding fold
"""
#Import the linear regression model
from sklearn.linear_model import LinearRegression
#Helped for cross validation 
from sklearn.cross_validation import KFold

# The columns we will use to predict target (the features?)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# Initialize transfer function
lr = LinearRegression()
# Generate cross validation folds for the titanic dataset. It should 
# return the row indices corresponding to the training test set.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1) # Lookup random_state in docs
# titanic.shape[0] would be the rows


predictions = [] # The predictions that will be made
for train, test in kf: # Basically a nested for loop
	# Predictors being used to train algorithm. Only take the rows in
	# the train folds?
	train_predictors = (titanic[predictors].iloc[train,:])
	# The target we are using to train the algorithm
	train_target = titanic["Survived"].iloc[train]
	# Training the algorithm with predictors and target
	lr.fit(train_predictors, train_target)
	# We can now make predictions on the test fold
	test_predictions = lr.predict(titanic[predictors].iloc[test,:])
	predictions.append(test_predictions) # Add predictions to end of array

"""
Evaluating the error! :D 
------------------------
First need to define error metric so we can figure out how accurate our 
model is. The error metric is a percentage of correct predictions. We
can use this same metric to evaluate our performance locally.

This metric will involve finding the number of values in predictions that 
are the exact same as their counterparts in titanic["Survived"], and Then
divided by the total number of passangers.

Before we can do this we must combine the 3 sets of predictions into one 
column. Since each set of predictions is numpy array, we should use a numpy
function to concatenate them into one.

"""

# The predictions are, so far, in 3 separate numpy arrays - concatenate 
# them into one.

# We concatenate them on the axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis = 0)

# Map prediction to outcome (only possible outcomes are 1 or 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print accuracy