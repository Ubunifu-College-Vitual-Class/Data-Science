# Using the advertising model attached , create a model that will predict whether or not that an internet user will click on an ad based on the features.

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Advertising.csv")

#encode categorical column
cols = ['Ad Topic Line', 'City', 'Country']
le = LabelEncoder(); 
df[cols] = df[cols].apply(le.fit_transform)

#extract features/attributes and target variable
X = df.drop(['Clicked on Ad','Timestamp'], axis = 1).values
y = df['Clicked on Ad'].values


#Split the data into training set and testing set using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train and fit a logistic regression model on the training set.
logReg = LogisticRegression()
logReg.fit(X_train, y_train)

#Now predict values for the testing data.
predictions = logReg.predict(X_test)

#Create a classification report for the model.
print(classification_report(y_test, predictions))

# #predict a new user if will click an add using a new point:    
#Daily Time Spent on Site	#Age	#Area Income	#Daily Internet Usage	#Ad Topic Line	#City	#Male	#Country
raw_point = [50,40,50000,200,'Centralized neutral neural-net','Michelleside',1,'Kenya'] # 1

#raw_point = [0.0,0,0,0,'Sample non existing category','Nairobi',1,'Kenya'] # 0


raw_point = le.fit_transform(raw_point) #encode using the same label encoder used in training
#make a prediction    
new_prediction = logReg.predict([raw_point]) 

if new_prediction == 1:
    print (f"The user is likely to click the advert ; {new_prediction}")
else:
    print (f"The user will not click the advert ;{new_prediction}")
