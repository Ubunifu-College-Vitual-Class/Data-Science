# You are a public health researcher interested in social factors that influence heart disease.
# You survey 500 towns and gather data on the percentage of people in each town who smoke, the percentage of people in each town who bike to work, and the percentage of people in each town who have heart disease.
# Illustrate the regression coefficient and its meaning to your model.
# Illustrate the standard error of the estimate ,  and the  p value.
# Dataset Link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/download?datasetVersionNumber=2
 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Convention alias for Seaborn
import os
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error 



# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Heart.csv")


variables = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='target', data=df).set(title=f'Regression plot of {var} and heart test result');
    
    
    
#calculate the correlation of the new variables, this time using Seaborn's heatmap()    
correlations = df.corr()
# annot=True displays the correlation values
sns.heatmap(correlations, annot=True).set(title='Heatmap of Test results x1,x2 & x3 Data - Pearson Correlations');



y = df['target']
#we have 3 columns instead of one.
x = df[variables]   

#divide our data into train and test sets.using the 42 as seed and 20% of our data for training
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42) 


#using the fit() method of the LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



# # #inspect the intercept 
print(regressor.intercept_)

# #retrieving the slope (which is also the coefficient of x) - shows what happens to the dependent variable whenever there is an increase (or decrease) of one unit of the independent variable
print(regressor.coef_)

# the four values are the coefficients for each of our features in the same order as we have them in our X data
feature_names = x.columns


#create a dataframe with our features as an index and our coefficients as column values
model_coefficients = regressor.coef_
coefficients_df = pd.DataFrame(data = model_coefficients, index = feature_names, columns = ['Coefficient value'])
print(coefficients_df)



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
print(f'Root mean squared error: {rmse:.2f}')
















































