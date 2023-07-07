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
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score 


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
X_train, X_test, y_train, y_test = train_test_split(x_feature_scaled, y,test_size=0.2,random_state=42)

#using the fit() method of the LinearRegression
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#making predictions
#pass the independent variables
y_pred = regressor.predict(X_test)

#compare them with the actual output values for y_test by organizing them in a DataFrame
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}') #A value of zero would indicate a perfect fit to the data.

score = r2_score(y_test,y_pred)#use the R2 score to get the accuracy of your model on a percentage scale, that is 0–100, just like in a classification model.
print("The accuracy of our model is {}%".format(round(score, 2) *100))
















































