

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Minisom library and module is used for performing Self Organizing Maps
from minisom import MiniSom #pip install MiniSom
import os
from sklearn.preprocessing import MinMaxScaler
from pylab import plot, axis, show, pcolor, colorbar, bone


# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Credit_Card_Applications.csv")


# Defining X variables for the input of SOM
x = df.iloc[:, 1:14].values
y = df.iloc[:, -1].values


# StandardScaler and MinMaxScaler are two common techniques for scaling numerical input variables for machine learning algorithms. 
# Scaling the data means transforming the values to a standard range, which can improve the performance and accuracy of some algorithms.
# The choice between MinMaxScaler and StandardScaler depends on the data distribution, the nature of the analysis, and the algorithm being used. 
# The main difference between StandardScaler and MinMaxScaler is how they transform the data:
# • StandardScaler removes the mean and scales the data to unit variance. It assumes that the data follows a normal distribution and uses the z-score formula to rescale the values: z = (x - mean) / std. 
# This means that each value will have a mean of zero and a standard deviation of one after scaling. 
# StandardScaler is useful when the data has outliers or different scales of measurement
# • MinMaxScaler rescales the data to a fixed range, typically between 0 and 1. 

sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)
X=x  #temp variable X

# Set the hyper parameters
som_grid_rows = 10
som_grid_columns = 10
iterations = 20000
sigma = 1
learning_rate = 0.5

# define SOM:
som = MiniSom(x = som_grid_rows, y = som_grid_columns, input_len=13, sigma=sigma, learning_rate=learning_rate)
# Initializing the weights
som.random_weights_init(x)
# Training
som.train_random(x, iterations)

# # Weights are:
# print(som._weights)
# # Shape of the weight are:
# print(som._weights.shape)
# # Returns the distance map from the weights:
# print(som.distance_map())

print(bone())
print(pcolor(som.distance_map().T))       # Distance map as background
print(colorbar()) #gives legend

markers = ['o', 's']                    # if the observation is fraud then red circular color or else green square
colors = ['r', 'g']

for i, x in enumerate(x):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# The markers used to distinguish frauds are:
# Red Circular is Class 0 as fraud customers
# Green Square is Class 1 as not fraud customers
# i is the index and x is the vector of each value and for each x first get the winning node
# The co-ordinates of the winning node are w[0] and w[1], 0.5 is added to center the marker
# s is a square and marker face color is the inside color of the marker

#There are some customers who don't have mapping above so those wouldn't be part of the segments..
mappings = som.win_map(X)
#print(mappings)
print(mappings.keys())#red circular from the heat ma
# Rem: With x = 10 and y = 10 as the respective number of rows and number of columns (dimensions), there will be 10* 10 meaning 100 segments.
print(len(mappings.keys()))
print(f"Out of the 100 segments: {len(mappings.keys())} segments have customers and other { 100 - len(mappings.keys()) } segments don't have any customers mapped to it.")

# Taking some of the red circular from the heat map and mapping as Frauds:
frauds = np.concatenate((mappings[(0,9)], mappings[(8,9)]), axis = 0)

print(frauds)
# the list of customers who are frauds:
fraud_customers = sc.inverse_transform(frauds)
print(pd.DataFrame(fraud_customers))


















