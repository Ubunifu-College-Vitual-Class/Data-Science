# A company manufactures an electronic device to be used in a very wide temperature range. 
# The company knows that increased temperature shortens the life time of the device, and a study is therefore performed in which the life time is determined as a function of temperature.
# The following data is found (CompanyDeviceManufacture.csv)

import numpy as np 
import pandas as pd
import seaborn as sns # Convention alias for Seaborn
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error  


df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\CompanyDeviceManufacture.csv")

print(df)
#or
# #initialize from a list
# data = [[10, 430], [20, 378], [30, 291], [40, 250], [50, 189], [60, 165], [70, 68],[80, 35], [90, 6]]  
# # Create the pandas DataFrame
# df = pd.DataFrame(data, columns=['Temperature', 'Lifetime'])


#A great way to explore relationships between variables is through Scatterplots
df.plot.scatter(x='Temperature', y='Lifetime', title='Scatterplot of temp(C) and lifetime(hrs)');

# #he corr() method calculates and displays the correlations between numerical variables in a DataFrame

print(df.corr())

# # #statistical summaries,
#print(df.describe())

#divide our data in two arrays - one for the dependent feature and one for the independent
    
y = df['Lifetime'].values.reshape(-1, 1)
x = df['Temperature'].values.reshape(-1, 1)

# print(X)  

SEED = 42 #random state hyperparameter in the train_test_split() function controls the shuffling process
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = SEED)


#Training a Linear Regression Model
regressor = LinearRegression()

# fit the line to our data
regressor.fit(X_train, y_train)

# #inspect the intercept 
# print(regressor.intercept_)

# #retrieving the slope (which is also the coefficient of x) - shows what happens to the dependent variable whenever there is an increase (or decrease) of one unit of the independent variable
# print(regressor.coef_)


#Making Predictions
electronic_device_lifetime = regressor.predict([[100]])
print(electronic_device_lifetime)


y_pred = regressor.predict(X_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

#A regression plot is useful to understand the linear relationship between two parameters. It creates a regression line in-between those parameters and then plots a scatter plot of those data points.
sns.regplot(x=x,y=y,ci=None,color ='red');

























