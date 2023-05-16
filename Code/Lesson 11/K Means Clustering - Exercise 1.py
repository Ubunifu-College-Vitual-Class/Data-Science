# Bob has started his own mobile company.
# He wants to give tough fight to big companies like Apple, Samsung etc.
# He does not know how to estimate price of mobiles his company creates.
# In this competitive mobile phone market you cannot simply assume things.
# To solve this problem he collects sales data of mobile phones of various companies.
# Bob wants to find out some relation between features of a mobile phone(e.g :- RAM, Internal Memory etc.) and its selling price. 
# But he is not so good at Machine Learning. So he needs your help to solve this problem.
# We do not have to predict actual price but a price range indicating how high the price is

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  



# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\MobilePhoneData.csv")


# select the relevant features
x = df[['battery_power','dual_sim','int_memory','n_cores','px_height', 'px_width','ram','touch_screen','wifi']].values

# scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x)


#5 clusters i.e k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)
kmeans.fit(x_train)  




#bob has the below parameters of his new phone
#battery_power #dual_sim #int_memory #n_cores #px_height #px_width #ram #touch_screen #wifi
bobs_phone = [1500,1,64,8,256,1024,156,1,1] #new point


#reshape data
# Reshaping means changing the shape of an array.
# The shape of an array is the number of elements in each dimension.
# By reshaping we can add or remove dimensions or change number of elements in each dimension.

reshaped_data = np.array([bobs_phone]).reshape(1, -1) 
#scale the new data
new_data_scaled = scaler.transform(reshaped_data)

# predict the cluster label for the new data point
cluster_label = kmeans.predict(new_data_scaled)[0]


# classify the phone based on the cluster label
if cluster_label == 0:
    print ("Bob's phone is on the Cheap cluster")
elif cluster_label == 1:
    print ("Bob's phone is on the Average cluster")
elif cluster_label == 2:
    print ("Bob's phone is on the Expensive cluster")
else:
    print ("Bob's phone is on the Premium phone cluster")
    
#Plotting Label 0 K-Means Clusters  



#filter rows of original data

 
#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(cluster_label)
 
#plotting the results:

plt.legend()
plt.show()







