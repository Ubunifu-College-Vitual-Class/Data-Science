# Does the higher cost of tuition translate into higher-paying jobs? 
# The table lists the top ten colleges based on mid-career salary and the associated yearly tuition costs. Construct a scatter plot of the data.
import pandas as pd
import numpy as np 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error  

# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Campus-tuition-fess-and-Career-salary.csv")

print(df)

sns.set_style = "whitegrid"
sns.histplot(x = 'Yearly Tuition', data = df, bins =30)

#Create a jointplot showing Mid-Career Salary (in thousands) versus Yearly Tuition.
sns.jointplot(x = 'Yearly Tuition', y= 'Mid-Career Salary (in thousands)', data = df, hue = "Mid-Career Salary (in thousands)")


X = df.drop(['Mid-Career Salary (in thousands)','School'], axis = 1)
y = df['Mid-Career Salary (in thousands)']

#Split the data into training set and testing set using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train and fit a logistic regression model on the training set.
linearReg = LinearRegression()
linearReg.fit(X_train, y_train)

#Now predict values for the testing data.
y_pred = linearReg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

#predicting mid-year salary of a school offering 200000
new_point_predictions = linearReg.predict([[200000]])
print(f'The mid-career salary predicted for tuition 200,000 is : {new_point_predictions[0]:.0f} (in thousands)')