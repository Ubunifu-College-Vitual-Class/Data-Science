# Importing required libraries
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Wisconsin Cancer.csv")
print(df.head())


# our feature variable and remove unnecessary columns
X = df.drop(["diagnosis","id","Unnamed: 32"],axis=1).values
# target variable
y = df.loc[:,"diagnosis"].values

# fit and transform our target variable
le = LabelEncoder()
y = le.fit_transform(y)


# split the dataset into training and test set
X_train,X_test,y_train,y_test=train_test_split(X, y,stratify=y,random_state=0)


# # create an object from this class
# rf = RandomForestClassifier(random_state = 0)
# # build our model
# rf.fit(X_train, y_train)


# # evaluate our model using the training and test set.
# y_train_pred = rf.predict(X_train)
# y_test_pred = rf.predict(X_test)

# #  look at accuracy scores for the training and test set
# rf_train = accuracy_score(y_train, y_train_pred)
# rf_test = accuracy_score(y_test, y_test_pred)

# # print these scores
# print(f'Random forest train/test accuracies: {rf_train: .3f}/{rf_test:.3f}')


rf = RandomForestClassifier(random_state = 42)

#a dictionary of hyperparameters to evaluate the param_grid argument. 
parameters = {'max_depth':[5,10,20],                           
              'n_estimators':[i for i in range(10, 100, 10)],  
              'min_samples_leaf':[i for i in range(1, 10)],     
              'criterion' :['gini', 'entropy'],               
              'max_features': ['auto', 'sqrt', 'log2']} 
       
# find the best parameters
clf = GridSearchCV(rf, parameters, n_jobs= -1)


# fit our model with the training set
clf.fit(X_train, y_train)

# to see the best parameters
#print(clf.best_params_)


y_train_pred=clf.predict(X_train)
y_test_pred=clf.predict(X_test)
rf_train = accuracy_score(y_train, y_train_pred)
rf_test = accuracy_score(y_test, y_test_pred)
print(f'Random forest train/test accuracies: {rf_train: .3f}/{rf_test:.3f}')
































