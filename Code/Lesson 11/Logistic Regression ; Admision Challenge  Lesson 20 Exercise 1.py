# Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams.
# You have historical data from previous applicants that you can use as a training set for logistic regression. 
# For each training example, you have the applicant’s scores on two exams and the admissions decision. 
# Your task is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams.

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Uni Admission Dataset.csv")

#extract features/attributes and target variable
X = df.drop(['y'], axis = 1).values
y = df['y'].values

#Split the data into training set and testing set using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train and fit a logistic regression model on the training set.
logReg = LogisticRegression()
logReg.fit(X_train, y_train)

#Now predict values for the testing data.
predictions = logReg.predict(X_test)

#Create a classification report for the model.
print(classification_report(y_test, predictions))

#predict using actual data
new_prediction = logReg.predict([[55,60]])


if new_prediction == 1:
    print (f"The student has passed the admission test ; {new_prediction}")
else:
    print (f"The student has failed the admission test ;{new_prediction}")
