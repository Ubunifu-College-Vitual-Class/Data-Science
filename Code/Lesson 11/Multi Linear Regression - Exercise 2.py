# The human resource at Nairobi County Council seeks to add more workers at the CBD. They conduct an aptitude test for 25 individuals, who all are promoted to the next test to 
# be carried out as  x1, x2 and x3.
# The aptitude test will be regarded as job proficiency of the individual.
# Below is a model to predict the expected aptitude required based on a single individual test results of 
# X1 = 112
# X2 = 119
# X3 = 106
# Dataset in CSV format (JobDataNairobiCounty) comes along this exercise.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Convention alias for Seaborn
import os
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error 


df = pd.read_excel(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\JobDataNairobiCounty.xlsx", sheet_name='Sheet1')

# #comparison of the statistics by rounding up the values to two decimals with the round() method, and transposing the table with the T property
# print(df.describe().round(2).T)


variables = ['x1', 'x2', 'x3']

for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='y', data=df).set(title=f'Regression plot of {var} and aptitude test result');
    
    
    
#calculate the correlation of the new variables, this time using Seaborn's heatmap()    
correlations = df.corr()
# annot=True displays the correlation values
sns.heatmap(correlations, annot=True).set(title='Heatmap of Test results x1,x2 & x3 Data - Pearson Correlations');



y = df['y']
#we have 3 columns instead of one.
x = df[['x1', 'x2','x3']]   

#divide our data into train and test sets.using the 42 as seed and 20% of our data for training
SEED = 0
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=SEED) 


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

#A regression plot is useful to understand the linear relationship between two parameters. It creates a regression line in-between those parameters and then plots a scatter plot of those data points.
#sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');

#x1 , #x2 , #x3
predictX = [112,119,106]
prediction = regressor.predict([predictX])
print(f'Aptitude prediction of a test with x1 = 112 , x2 = 119 & x3 = 106 is {prediction}')














































