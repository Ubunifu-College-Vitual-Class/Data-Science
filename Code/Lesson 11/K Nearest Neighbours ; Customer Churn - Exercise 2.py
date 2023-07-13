# Download the dataset from the following link: Customer Churn Dataset (https://huggingface.co/datasets/d0r1h/customer_churn/raw/main/churn.csv ).
# Load the dataset into a pandas Data Frame.
# Pre-process the data by handling missing values, encoding categorical variables (if any), and performing any necessary feature scaling or transformations.
# Split the data into training and testing sets.
# Create a KNN classifier using scikit-learn's KNeighborsClassifier class.
# Fit the model to the training data.
# Choose an appropriate number of neighbours (k) by experimenting with different values (elbow method etc) and evaluating the model's performance on the testing data.
# Evaluate the model's performance by calculating metrics such as accuracy, precision, recall, and F1-score on the testing data.
# Interpret the results and discuss the trade-offs between different performance metrics.
# Experiment with different pre-processing techniques, feature selections, or hyper parameter tuning to improve the model's performance.
# Discuss the limitations and assumptions of the KNN algorithm for this particular problem.

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # Convention alias for Seaborn
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np  


#importing datasets 
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Customers Churn.csv")

df['region_category'] = df['region_category'].fillna(df['region_category'].mode()[0])

#replace values with  ? symbol
df['medium_of_operation'].replace(['?'], '')
df['joined_through_referral'].replace(['?'], 'No')
df['avg_frequency_login_days'].replace('Error',0,inplace=True)






#impute nan using mode
df['region_category'] = df['region_category'].fillna(df['region_category'].mode()[0])
#df['joined_through_referral'] = df['joined_through_referral'].fillna(df['joined_through_referral'].mode()[0])
df['medium_of_operation'] = df['medium_of_operation'].fillna(df['medium_of_operation'].mode()[0])
df['points_in_wallet'] = df['points_in_wallet'].fillna(df['points_in_wallet'].mode()[0])
df['region_category'] = df['region_category'].fillna(df['region_category'].mode()[0])
df['avg_frequency_login_days'] = df['avg_frequency_login_days'].fillna(df['avg_frequency_login_days'].mode()[0])



#encode categorical columns
cols = ['gender', 'region_category', 'membership_category','joined_through_referral','preferred_offer_types','medium_of_operation','internet_option','used_special_discount','offer_application_preference','past_complaint','complaint_status','feedback']
le = LabelEncoder(); 
df[cols] = df[cols].apply(le.fit_transform)

#assigning the attributes and tagret
x =  df[['age','gender','region_category','membership_category','joined_through_referral','preferred_offer_types','medium_of_operation','internet_option','days_since_last_login','avg_time_spent','avg_transaction_value','avg_frequency_login_days','points_in_wallet','used_special_discount','offer_application_preference','past_complaint','complaint_status','feedback']] .values  #dataset contains lots of information but the Estimated Salary and Age we will consider for the independent variable
y = df['churn_risk_score'].values

# Splitting the dataset into training and test set.  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0) 

# elbow method
# generate a plot showing the sum of squared distances for each K value. The "elbow" in the plot represents the optimal K value
k_values = range(1, 11)
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0) # Set n_init explicitly
    kmeans.fit(x_train)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_values, inertia_values, 'bx-')
plt.xlabel('K')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method')
plt.show() # visually identify the optimal k value

optimal_k = 3

# fitting K-NN classifier to the training set  
classifier= KNeighborsClassifier(n_neighbors=optimal_k, metric='minkowski', p=2 )  # metric='minkowski' parameter specifies the distance metric used to calculate the distances between samples (is a generalization of other distance metrics such as the Euclidean distance and the Manhattan distance)
classifier.fit(x_train, y_train)

#predicting the test set result  
y_pred = classifier.predict(x_test)  

#creating the Confusion matrix
cm = confusion_matrix(y_test, y_pred) 

# to see how accurate our algorithm is
# print(cm)

#compare them with the actual output values for y_test by organizing them in a DataFrame
results = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred)})
print(results)

print(classification_report(y_test, y_pred))
#accuracy - how much percentage of cases that you have predicted correctly
#recall - how much percentage of real positive cases are correctly identified
#precision - among the cases predicted to be positive, how much percentage of them are really positive
#f1-score - harmonic mean of recall and precision
#harmonic mean is a type of average that is calculated by dividing the number of values in a data series by the sum of the reciprocals

accuracy = accuracy_score(y_test, y_pred)#measure of how well the classifier predicts the correct class labels for new data points.
#It is calculated as the ratio of the number of correctly predicted data points to the total number of data points.
print(f'K-NN model accuracy : {accuracy*100:.0f} %')

# Over 90% - Very good.
# Between 70% and 90% - Good.
# Between 60% and 70% - OK.
# Below 60% - Poor.























