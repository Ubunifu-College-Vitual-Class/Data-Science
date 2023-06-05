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

# Dropping missing records
df = df.dropna()
# print(len(df))


# # Plotting our penguin species features
# pairplot(df, hue='species')
# plt.show()


# Splitting our data
X = df[['culmen_length_mm', 'culmen_depth_mm']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)


# Building and training our model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Making predictions with our data
predictions = clf.predict(X_test)

print(predictions[:5]) # display the first five elements (rows) of the predictions array

# # Returns:['Gentoo' 'Chinstrap' 'Gentoo' 'Adelie' 'Gentoo']

# # Visualizing the linear function for our SVM classifier

w = clf.coef_[0] #represents the weights assigned to the features in the model
b = clf.intercept_[0] # the "intercept_" attribute represents the constant term in the linear equation
x_visual = np.linspace(32,57) # generates a sequence of evenly spaced numbers between 32 and 57 using the numpy "linspace" function
y_visual = -(w[0] / w[1]) * x_visual - b / w[1] # this line calculates the corresponding y-axis values for visualization based on the linear classifier's coefficients and intercept.

scatterplot(data = X_train, x='culmen_length_mm', y='culmen_depth_mm', hue=y_train)
plt.plot(x_visual, y_visual)
plt.show()


# # Testing the accuracy of our model
print(accuracy_score(y_test, predictions))

                                                                                                                                                                   
# new data point prediction

new_penguin_dimensions_one = pd.DataFrame([[49.9 ,16.1]], columns = ['culmen_length_mm', 'culmen_depth_mm']) #Gentoo
new_penguin_dimensions_two = pd.DataFrame([[48.5,17.5]], columns = ['culmen_length_mm', 'culmen_depth_mm']) #Chinstrap


new_penguin_dimensions_random = pd.DataFrame([[1.5,7.5]], columns = ['culmen_length_mm', 'culmen_depth_mm'])

predictede = clf.predict(new_penguin_dimensions_random) 

print(f'Predicted the species as : {predictede[0]}')









