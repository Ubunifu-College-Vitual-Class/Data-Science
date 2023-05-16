import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Petrol_Consumption.csv")

#first five records of the dataset
print(df.head())

#divide data into labels and attributes
x = df.drop('Petrol_Consumption', axis=1)
y = df['Petrol_Consumption']

#divide our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0) 

#train the tree
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#make predictions on the test set, ues the predict method
y_pred = regressor.predict(X_test)

#compare some of our predicted values with the actual values and see how accurate we were
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
#print(df)

#evaluate performance of the regression algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))