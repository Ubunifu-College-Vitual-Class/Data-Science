# Download the dataset from the following link: House Prices Dataset (https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/download?datasetVersionNumber=1 ).
# Load the dataset into a pandas DataFrame.
# Preprocess the data by handling missing values, encoding categorical variables (if any), and performing any necessary feature scaling or transformations.
# Split the data into training and testing sets.
# Create a multilinear regression model using scikit-learn's LinearRegression class.
# Fit the model to the training data.
# Evaluate the model's performance by calculating metrics such as mean squared error (MSE), mean absolute error (MAE), and R-squared score on the testing data.
# Interpret the model coefficients to understand the impact of each feature on house prices.
# Experiment with different preprocessing techniques, feature selections, or hyperparameter tuning to improve the model's performance.
# Discuss the results, including any limitations or assumptions of the model. 

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Housing.csv")

#encode categorical columns
cols = ['mainroad', 'guestroom', 'basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
le = LabelEncoder(); 
df[cols] = df[cols].apply(le.fit_transform)
print(df)

y = df['price']
x = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']] 

# feature scale the data
scaler = StandardScaler()
x_feature_scaled = scaler.fit_transform(x)  

# #divide our data into train and test sets.using the 42 as seed and 20% of our data for training
X_train, X_test, y_train, y_test = train_test_split(x_feature_scaled, y,test_size=0.2,random_state=0)

#using the fit() method of the LinearRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

#making predictions
#pass the independent variables
y_pred = regressor.predict(X_test)

#compare them with the actual output values for y_test by organizing them in a DataFrame
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
#Create a classification report for the model.
print(classification_report(y_test, y_pred,zero_division=1)) # zero_division parameter to 1, meaning that when there are no predicted samples for a label, the precision and F-score will be set to 0/1 = 0 instead of generating a warning.

# Extract the accuracy from the report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy))
















































