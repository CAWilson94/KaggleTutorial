""" 
Going through Kaggle tutorial 
(uses interactive code boxes)

"""
import csv as csv
import numpy as mp

csvFile = csv.reader(open('train.csv','rb'))
# We have a header line, so need to skip this!
headerSkip = csvFile.next()
# Shove the data into array (would normally use bunch object)
data = []

# Could normally use the pandas library to do this automatically ...
for row in csvFile     # Loop through each row
	data.append(row)   # Add each row to the data variable
data = np.array(data)  # convert from List to Array