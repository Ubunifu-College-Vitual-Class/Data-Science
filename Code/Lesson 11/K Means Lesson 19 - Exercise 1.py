# Your preferred workout is jogging and, since you're extremely data-inclined, you make sure to keep track of your performance. 
# So you end up compiling a dataset similar to this :
# Datasets\\Jogging.csv   
# It consists of the date you went jogging, the total distance run (Km), duration (Min) and the number of days since your last workout.
# Each row in your dataset contains the attributes or features of each workout session.
# Identify workout sessions that are similar to each other so you can have a better understanding of your overall performance and get new ideas on how to improve it.


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  



# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Jogging.csv")


# select the relevant features
x = df[['distance_km','duration_min','delta_last_workout']].values

# scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x)

#finding optimal number of clusters using the elbow method 
wcss_list= []  #Initializing the list for the values of WCSS 


# Within-cluster sums of squares. This is a measure of how compact the clusters are in a clustering algorithm, such as K-means.
# It is calculated as the sum of the squared distances between each data point and its cluster centroid.
# The lower the WCSS, the more similar the data points are within each cluster.
# Minimizing WCSS is equivalent to maximizing the distance between clusters


#Using for loop for iterations from 1 to 10.  
for i in range(1, 7):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(x_train)  
    wcss_list.append(kmeans.inertia_)  
plt.plot(range(1, 7), wcss_list)  
plt.title('The Elobw Method Graph')  
plt.xlabel('Number of clusters(k)')  
plt.ylabel('wcss_list')  
plt.show()  


#5 clusters i.e k=5
kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)
df['cluster'] = kmeans.fit_predict(x_train)  #return the labels



# filtered_label0 = df[label == 0]
label = df['cluster']

label_0 = df[label == 0]
label_1 = df[label == 1]
label_2 = df[label == 2]
label_3 = df[label == 3]
label_4 = df[label == 4]

cols = df.columns
plt.scatter(label_0[cols[0]], label_0[cols[1]], color = 'red')
plt.scatter(label_1[cols[0]], label_1[cols[1]], color = 'black')
plt.scatter(label_2[cols[0]], label_2[cols[1]], color = 'green')
plt.scatter(label_3[cols[0]], label_3[cols[1]], color = 'blue')
plt.scatter(label_4[cols[0]], label_4[cols[1]], color = 'yellow')
plt.show()





