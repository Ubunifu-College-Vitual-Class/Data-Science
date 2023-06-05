# Using the dataset Link : https://www.kaggle.com/code/nilslefeuvre/credit-card-fraud-detection-using-ml-dl/input?select=creditcard.csv  , you are issued with a classification task to predict whether a bank transaction is fraudulent or not based on certain features.
# 1). Create your model
# 2). Print the Accuracy , Precision , Recall and explain them in the above context
# 3). Visualize
# 4). Do a prediction on a certain new point



# Importing required libraries
from seaborn import load_dataset, pairplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np
from seaborn import scatterplot


#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Creditcard.csv")

# Dropping missing records
df = df.dropna()

df.sample(n = 50, random_state = 20)


# Splitting our data
X = df[['V1','V2']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)


# Building and training our model

# to fit non-linear data, you can use kernels like 'poly', 'rbf', or 'sigmoid'. 
# RBF (Gaussian) kernel with the SVC class from scikit-learn:

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Making predictions with our data
predictions = clf.predict(X_test)


# Visualizing the linear function for our SVM classifier

w = clf.dual_coef_ [0] #represents the weights assigned to the features in the model
b = clf.intercept_[0] # the "intercept_" attribute represents the constant term in the linear equation
x_visual = np.linspace(32,57) # generates a sequence of evenly spaced numbers between 32 and 57 using the numpy "linspace" function
y_visual = -(w[0] / w[1]) * x_visual - b / w[1] # this line calculates the corresponding y-axis values for visualization based on the linear classifier's coefficients and intercept.



variables = ['V1','V2']
for var in variables:
    scatterplot(data = X_train, x=var, y=y, hue=y_train)
    plt.plot(x_visual, y_visual)
    plt.show()


# scatterplot(data = X_train, x='V1', y='V2', hue=y_train)
# plt.plot(x_visual, y_visual)
# plt.show()


# # Testing the accuracy of our model
print(accuracy_score(y_test, predictions))

                                                                                                                                                                   
# # new data point prediction

new_point = pd.DataFrame([[-1.570300537,1.8805901]], columns = ['V1','V2'])

predicted = clf.predict(new_point) 
print(f'The bank transaction is : {predicted[0]}')










