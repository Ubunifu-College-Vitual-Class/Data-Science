import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error   


# Using the Kaggle dataset : https://www.kaggle.com/datasets/ruiromanini/mtcars/download?datasetVersionNumber=1 
# 1). Display the full data set
# 2). Shape of the training dataset
# 3). Predict automobile per gallon (mpg) for a vehicle with 200hp
# 4). Display your mobile accuracy metrics
# 5). Visualize your model with matplot lib





# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Automobile_mtcars.csv")

# 1). Display the full data set
print(df)  #or use print(df.head())


# 2). Shape of the training dataset
shape = df.shape
print('DataFrame Shape :', shape)
print('Number of rows :', shape[0])
print('Number of columns :', shape[1])


y = df['mpg'].values
x = df['hp'].values


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature Scaling 
st_x= StandardScaler() #data normalization , performed during the data preprocessing   
x_train= st_x.fit_transform(X_train.reshape(-1, 1))    
x_test= st_x.transform(X_test.reshape(-1, 1)) 


#Training a Linear Regression Model
regressor = LinearRegression()

# fit the line to our data
regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)

#Display test dataset
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

horse_power = st_x.transform([[200]])
# 3). Predict automobile per gallon (mpg) for a vehicle with 200hp
mpg_predicted = regressor.predict([[1]])

print(f"Automobile per gallon for a 200hp vehicle is {mpg_predicted[0]:.0f}")

# 5). Visualize your model with matplot lib
plt.scatter(X_train, y_train,color='g')
plt.plot(X_test, y_pred,color='k') 
plt.show()






























