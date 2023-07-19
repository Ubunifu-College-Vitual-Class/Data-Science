# Kenya has experienced a significant transformation in its financial landscape with the introduction of mobile money services.
#  This practical exercise examines the impact of mobile money on financial inclusion in Kenya, focusing on the adoption, usage, and benefits of mobile money platforms.
# The dataset is included in this exercise (Mobile Money on Financial Inclusion in Kenya) , where collected data on the following variables for a sample of individuals:
# Mobile Money Usage (number of mobile money transactions per month)
# Bank Account Ownership (binary variable indicating whether the individual has a bank account)
# Savings (amount of savings in Kenyan Shillings)
# Credit Usage (binary variable indicating whether the individual has used mobile money credit services)


# Implement a K-Means model on the dataset with an assumption of (k=2)
# Cluster 0:
# This cluster represents individuals who tend to have lower mobile money usage, don't own a bank account, have lower savings, and do not use credit services. They may be categorized as individuals who are less engaged in financial transactions and services. The characteristics of this cluster suggest a lower level of financial inclusion.

# Cluster 1:
# This cluster represents individuals who have higher mobile money usage, own a bank account, have higher savings, and use credit services. They demonstrate higher engagement in financial transactions and services. The characteristics of this cluster suggest a higher level of financial inclusion.


# Visualization using a scatter plot helps us understand the clustering results and how the data points are grouped based on their assigned cluster labels.
# The x-axis to represents the 'Mobile Money Usage‘.
# Y-axis represents the 'Savings' variable, indicating the amount of savings in Kenyan Shillings.

# To build and run on docker :
# docker buildx build -t <image-name> .  i.e docker buildx build -t k-means-financial-inclusion .   
# docker run --name k_means_fin_inclusion_container <image-name> i.e  docker run -v Container-Disk:/home/Container-Disk --name k_means_fin_inclusion_container k-means-financial-inclusion
#
# below is pushing to docker hub 
# docker build -t mugambidemundia/data-science:k-means-financial-inclusion .  ( docker build -t repository_name:tag . )
# docker login
# docker tag k-means-financial-inclusion mugambidemundia/data-science:k-means-financial-inclusion ( docker tag local_image repository_name:tag ) # tag the image with the Docker Hub repository
# docker push mugambidemundia/data-science:k-means-financial-inclusion ( docker push repository_name:tag )
#



import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np  

# importing the dataset  
df = pd.read_csv("Datasets/Mobile Money on Financial Inclusion in Kenya.csv")


# select the relevant features
x = df[['Mobile Money Usage','Bank Account Ownership','Savings','Credit Usage']].values

# scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x)

# clusters i.e k=2
kmeans = KMeans(n_clusters=2, init='k-means++', random_state= 42, n_init=10)
df['Cluster'] = kmeans.fit_predict(x_train)  #return the labels

print(df)

# visualize the distribution with cluster labels
plt.scatter(x_train[:, 0], x_train[:, 2], c=df['Cluster'], cmap='viridis') # colormap (cmap) is chosen as 'viridis' to provide a distinct color for each cluster
plt.xlabel('Mobile Money Usage')
plt.ylabel('Savings') #  Indicate the amount of savings in Kenyan Shillings.
plt.title('Distribution with Clusters')
plt.colorbar(label='Cluster')
plt.savefig('/home/Container-Disk/k-means-plot.png')  # Save the plot to an image file
plt.show()






