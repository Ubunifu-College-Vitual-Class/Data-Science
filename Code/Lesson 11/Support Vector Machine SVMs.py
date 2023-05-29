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
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Penguins.csv")
print(df.head())

# Dropping missing records
df = df.dropna()
print(len(df))


# Plotting our penguin species features
pairplot(df, hue='species')
plt.show()


# Splitting our data
X = df[['culmen_length_mm', 'culmen_depth_mm']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)


# Building and training our model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Making predictions with our data
predictions = clf.predict(X_test)
print(predictions[:5])

# Returns:['Gentoo' 'Chinstrap' 'Gentoo' 'Adelie' 'Gentoo']


# Visualizing the linear function for our SVM classifier

w = clf.coef_[0]
b = clf.intercept_[0]
x_visual = np.linspace(32,57)
y_visual = -(w[0] / w[1]) * x_visual - b / w[1]

scatterplot(data = X_train, x='culmen_length_mm', y='culmen_depth_mm', hue=y_train)
plt.plot(x_visual, y_visual)
plt.show()


# Testing the accuracy of our model
print(accuracy_score(y_test, predictions))






