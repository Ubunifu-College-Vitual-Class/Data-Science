import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Advertising.csv")

sns.set_style = "whitegrid"
sns.histplot(x = 'Age', data = df, bins =30)

#Create a jointplot showing Area Income versus Age.
sns.jointplot(x = 'Age', y= 'Area Income', data = df, hue = "Clicked on Ad")


#Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
sns.jointplot(kind = 'kde', x= "Area Income", y = "Age", data = df, hue = "Clicked on Ad", cmap = "Blues")


#Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(x = "Daily Time Spent on Site", y = "Daily Internet Usage", hue = "Clicked on Ad", data = df, cmap = "Green")



X = df.drop(['Clicked on Ad','Ad Topic Line', 'City', 'Country','Timestamp'], axis = 1)
y = df['Clicked on Ad']

#Split the data into training set and testing set using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


#Train and fit a logistic regression model on the training set.
logReg = LogisticRegression()
logReg.fit(X_train, y_train)

#Now predict values for the testing data.
predictions = logReg.predict(X_test)

#Create a classification report for the model.
print(classification_report(y_test, predictions))

