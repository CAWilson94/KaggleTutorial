""" 
Going through Kaggle tutorial 
(uses interactive code boxes)

"""
import csv as csv
import numpy as mp
import pandas as pd

titanic = pd.read_csv("train.csv")
print (titanic.head(5)) # Print first 5 rows of csv
print (titanic.describe()) # Age only shows 714 rows, others are 891. Need to clean data!

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median()) # Fill in empty cells with median of all 

print (titanic.describe())