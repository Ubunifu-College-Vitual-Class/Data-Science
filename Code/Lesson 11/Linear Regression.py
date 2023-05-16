import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error  


df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Student_Scores.csv")


#A great way to explore relationships between variables is through Scatterplots
df.plot.scatter(x='Hours', y='Scores', title='Scatterplot of hours and scores percentages');

# #he corr() method calculates and displays the correlations between numerical variables in a DataFrame

# print(df.corr())

# # #statistical summaries,
# print(df.describe())





#divide our data in two arrays - one for the dependent feature and one for the independent
    
y = df['Scores'].values.reshape(-1, 1)
x = df['Hours'].values.reshape(-1, 1)

# print(X)  

SEED = 42
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

# def calc(slope, intercept, hours):
#     return slope*hours+intercept
# score = calc(regressor.coef_, regressor.intercept_, 2.5)

score = regressor.predict([[1]])
print(score)


y_pred = regressor.predict(X_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

























