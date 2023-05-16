import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix



#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Bill_authentication.csv")

#number of rows and columns in our dataset
print(df.shape)

#first five records of the dataset
print(df.head())

#divide data into attributes and labels
x = df.drop('Class', axis=1)
y = df['Class']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

#train the algorithm
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#make predictions on the test data
y_pred = classifier.predict(X_test)

#to see how accurate our algorithm is
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))