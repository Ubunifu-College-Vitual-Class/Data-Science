# In Kenya, as in many other regions with significant wildlife populations, there is an ongoing challenge of wildlife-human conflict. 
# As human settlements expand into areas adjacent to national parks and wildlife reserves, encounters between wildlife and humans become more frequent. 
# These conflicts can lead to crop damage, livestock losses, and even human casualties, posing a threat to both wildlife conservation and human livelihoods.
# Dataset: wildlife_conflict_dataset.csv (attached)

# Using a Support Vector Machine (SVM) for this problem 
# Try and help to predict potential conflict zones and implement targeted mitigation strategies. 
# Data Collection: Gather data on historical wildlife sightings, human population density, land usage patterns, and reported incidents of wildlife-human conflicts in and around national parks and wildlife reserves.
# Data Pre-processing: Clean and pre-process the collected data, handling missing values and ensuring proper formatting for SVM input.
# Feature Engineering: Extract relevant features from the data that might influence wildlife movement and conflict occurrence, such as proximity to water sources, vegetation density, and distance to human settlements.
# Training the SVM: Utilize the pre-processed data to train the SVM. The algorithm will learn to classify areas into two categories: conflict-prone zones and non-conflict zones based on the provided features.
# Prediction: Once the SVM is trained, use it to predict potential conflict zones in areas where data is available but conflict incidents have not yet occurred. This will help in identifying regions that are likely to experience future conflicts.
# Mitigation Strategies: Based on the SVM predictions, implement targeted mitigation strategies in the identified conflict-prone zones. These strategies could include constructing barriers, developing early warning systems, promoting community awareness, and implementing measures to reduce human-wildlife interactions.
# Monitoring and Evaluation: Continuously collect new data on conflict incidents and monitor the effectiveness of the implemented mitigation strategies. Use this data to retrain the SVM periodically and improve its accuracy.


# below is pushing to docker hub 

# docker build -t mugambidemundia/data-science:svm-human-wildlife-confict .  ( docker build -t repository_name:tag . )

# docker login
# docker push mugambidemundia/data-science:svm-human-wildlife-confict ( docker push repository_name:tag )

# 

# docker run --name svm-human-wildlife-confict_container <image-name> i.e  docker run -v Container-Disk:/home/Container-Disk --name svm-human-wildlife-confict_container -t mugambidemundia/data-science:svm-human-wildlife-confict

# to remove : docker rm svm-human-wildlife-confict_container


# Importing required libraries

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np
from seaborn import scatterplot
from sklearn.preprocessing import MinMaxScaler # or StandardScaler


#read our CSV data file
df = pd.read_csv("Datasets/wildlife_conflict_dataset.csv")

# Dropping missing records
df = df.dropna()

df = df.head(15000)

# df.sample(n = 50, random_state = 20)


# Splitting our data
variables = ['Distance_to_Park','Proximity_to_Water','Vegetation_Density','Human_Population_Density']
X = df[variables]
y = df['Reported_Conflict_Incidents']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# initialize the MinMaxScaler
scaler = MinMaxScaler()

# fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building and training our model
# to fit non-linear data, you can use kernels like 'poly', 'rbf', or 'sigmoid'. 
# RBF (Gaussian) kernel with the SVC class from scikit-learn:
    
    
# linear Kernel: The linear kernel is best suited for linearly separable data. 
# It is computationally efficient and works well when the classes can be separated by a straight line or plane.

# polynomial Kernel: The polynomial kernel is useful when the decision boundary is not a straight line or plane but has curved shapes.
# It introduces non-linearity by transforming the features into higher dimensions.

# radial Basis Function (RBF) Kernel: The RBF kernel is a popular choice as it can handle non-linear data effectively. 
# It uses a Gaussian function to create complex decision boundaries, making it suitable for a wide range of problems.
# sigmoid Kernel: The sigmoid kernel can be used for non-linear classification tasks and is often used in neural network-inspired SVMs. 
# However, it may not perform as well as the RBF kernel in many cases.


# Building and training our model with poly kernel
clf = SVC(kernel='linear')
clf.fit(X_train_scaled, y_train)

# Making predictions with our data
predictions = clf.predict(X_test_scaled)


# visualizing the hyperplane for each pair of features
for i in range(len(variables)):
    for j in range(i + 1, len(variables)):
        feature_x = X_train_scaled[:, i]
        feature_y = X_train_scaled[:, j]

        plt.figure(figsize=(8, 6))
        scatterplot(data=X_train, x=variables[i], y=variables[j], hue=y_train)

        # plot the decision boundary (hyperplane)
        w = clf.coef_[0]
        b = clf.intercept_[0]
        x_visual = np.linspace(min(feature_x), max(feature_x), 100)
        y_visual = -(w[i] / w[j]) * x_visual - b / w[j]
        plt.plot(x_visual, y_visual, 'k-', label='Decision Boundary (Hyperplane)')

        plt.xlabel(variables[i])
        plt.ylabel(variables[j])
        plt.title(f'SVM Hyperplane Visualization for {variables[i]} vs {variables[j]}')
        plt.legend()
        plt.savefig('/home/Container-Disk/svm-human-wildlife-confict'+str(i)+'.png') 
        plt.show()
        
        
# # # Testing the accuracy of our model
accuracy = accuracy_score(y_test, predictions) * 100
print(f"SVM Model Accuracy: {accuracy:.2f}%")











