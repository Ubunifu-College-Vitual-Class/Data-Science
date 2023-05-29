import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Sent alongside this slide is a Kidney.csv that contains data recorded after 401 patient, 
# ckd is recorded for patients that were observed to have chronic kidney disease and notckd for those that did not have chronic kidney disease.
# 1). Create a suitable machine learning model.
# 2). Display the accuracy statistics for your model.
# 3). Predict a kidney infection for a patient with the following tests 

# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Kidney_disease.csv")

# Encode labels of multiple columns at once
cols = ['rbc', 'pc', 'pcc', 'ba', 'htn','dm','cad','appet','pe','ane','classification']

le = LabelEncoder()
df[cols] = df[cols].apply(le.fit_transform)

#prepare nan values
# Impute Mean in nan columns

df['age'].fillna(df['age'].mean(), inplace=True)
df['bp'].fillna(df['bp'].mean(), inplace=True)
df['sg'].fillna(df['sg'].mean(), inplace=True)
df['al'].fillna(df['al'].mean(), inplace=True)
df['su'].fillna(df['su'].mean(), inplace=True)
df['rbc'].fillna(df['rbc'].mean(), inplace=True)
df['pc'].fillna(df['pc'].mean(), inplace=True)
df['pcc'].fillna(df['pcc'].mean(), inplace=True)
df['ba'].fillna(df['ba'].mean(), inplace=True)
df['bgr'].fillna(df['bgr'].mean(), inplace=True)
df['bu'].fillna(df['bu'].mean(), inplace=True)
df['sc'].fillna(df['sc'].mean(), inplace=True)
df['sod'].fillna(df['sod'].mean(), inplace=True)
df['pot'].fillna(df['pot'].mean(), inplace=True)
df['hemo'].fillna(df['hemo'].mean(), inplace=True)

df['pcv'].fillna(method='ffill', inplace=True)#forward fill or backfill by ffill and bfill. 
df['wc'].fillna(method='ffill', inplace=True)
df['rc'].fillna(method='ffill', inplace=True)

df['htn'].fillna(df['htn'].mean(), inplace=True)
df['dm'].fillna(df['dm'].mean(), inplace=True)
df['cad'].fillna(df['cad'].mean(), inplace=True)
df['appet'].fillna(df['appet'].mean(), inplace=True)
df['pe'].fillna(df['pe'].mean(), inplace=True)
df['ane'].fillna(df['ane'].mean(), inplace=True)

#contains bad data i.e ?
df.at[66,'pcv']=0
df.at[67,'pcv']=0
df.at[162,'rc']=0
df.at[164,'rc']=0
df.at[185,'wc']=0
df.at[186,'wc']=0


#features 
x = df.drop(['classification','id'], axis=1).values
#target variable = has diabetes
y = df['classification'].values


#split the data into training and testing sets
#, 80/20% to 90/10% for large datasets;small dimensional datasets,  60/40% to 70/30%.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)# 42 produce the same results across a different run.

#feature Scaling 
st_x= StandardScaler() #data normalization , performed during the data preprocessing   
x_train= st_x.fit_transform(X_train)    
x_test= st_x.transform(X_test) 

#train the algorithm
regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)

#make predictions on the test data
y_pred = regressor.predict(x_test)


#to see how accurate our algorithm is
#evaluate performance of the regression algorithm
print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
# RMSE values between 0.2 and 0.5 shows that the model can relatively predict the data accurately
#The lower the RMSE, the better the model and its predictions.


#age #bp	#sg	#al	#su	#rbc #pc	#pcc	#ba	#bgr	#bu	#sc	#sod	#pot	#hemo	#pcv	#wc	#rc	#htn	#dm	#cad	#appet	#pe	#ane
predictKidneyDisease = [48.0,80.0,1.02,1.0,0.0,'normal','normal','notpresent','notpresent',121.0,36.0,1.2,10,4.0,15.4,44,7800,5.2,'yes','yes','no','good','no','no']
encodeData = le.fit_transform(predictKidneyDisease)#use same encode in training
kidneydidrisk = regressor.predict([encodeData])

if kidneydidrisk[0] == 0:
    print(f"\nThe patient is likely to have a chronic kidney disease (ckd:{kidneydidrisk[0]})")
else:
    print(f"\nThe patient is safe and not likely to have a chronic kidney disease (notckd:{kidneydidrisk[0]})")




















