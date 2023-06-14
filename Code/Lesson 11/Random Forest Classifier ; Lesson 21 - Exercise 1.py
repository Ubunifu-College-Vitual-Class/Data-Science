# Importing required libraries
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\melb_data.csv")
# print(df.head())

# drop rows emtpy YearBuilt 
df.dropna(subset = ['YearBuilt'], inplace = True) 

#Display missing values information
# missing_values = df.isna().sum().sort_values(ascending=True)
# print(missing_values);



#encode categorical column
cols = ['Suburb', 'Rooms', 'Type','Method','SellerG','Distance','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','CouncilArea','Regionname']
le = LabelEncoder(); 
df[cols] = df[cols].apply(le.fit_transform)

#impute nan using the most occurence
df['Car'].fillna(df['Car'].mode(), inplace=True)
df['CouncilArea'].fillna(df['CouncilArea'].mode(), inplace=True) 
df['BuildingArea'].fillna(df['BuildingArea'].mode(), inplace=True)


df = df.sample(1000)

# our feature variable and remove unnecessary columns
X = df.drop(["Price","Date","Address","Postcode","Lattitude","Longtitude","Propertycount"],axis=1).values
# target variable
y = df.loc[:,"Price"].values


# split the dataset into training and test set
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2,random_state=0)

# # Initialize the Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0) # initialized with 100 estimators (decision trees) and a random state of 42

# # Train the classifier
# rf_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = rf_classifier.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)


# # evaluate our model using the training and test set.
# y_train_pred = rf_classifier.predict(X_train)
# y_test_pred = rf_classifier.predict(X_test)

# #  look at accuracy scores for the training and test set
# rf_train = accuracy_score(y_train, y_train_pred)
# rf_test = accuracy_score(y_test, y_test_pred)
# # print these scores
# print(f'Random forest train/test accuracies: {rf_train: .3f}/{rf_test:.3f}')
# print('Accuracy: %.3f' % (accuracy*100)) # a higher accuracy score generally indicates better performance.


# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# 'n_estimators':  number of decision trees (estimators) 
# 'max_depth':  controls the maximum depth of each decision tree in the random forest. A higher max_depth allows the trees to have more splits and potentially capture more complex patterns in the data. 
# 'min_samples_split': This parameter specifies the minimum number of samples required to split an internal node of a decision tree.
# 'min_samples_leaf': This parameter represents the minimum number of samples required to be in a leaf node (a terminal node) of a decision tree.

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=0)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(rf_classifier, param_grid, cv=2)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the classifier with the best hyperparameters
best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)
best_rf_classifier.fit(X_train, y_train)

# Make predictions on the test set using the best classifier
y_pred = best_rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Evaluate the model using the training and test sets
y_train_pred = best_rf_classifier.predict(X_train)
y_test_pred = best_rf_classifier.predict(X_test)

# Look at accuracy scores for the training and test sets
rf_train = accuracy_score(y_train, y_train_pred)
rf_test = accuracy_score(y_test, y_test_pred)

# Print the scores
print(f'Random forest train/test accuracies: {rf_train:.3f}/{rf_test:.3f}')
print('Best hyperparameters:', best_params)
print('Accuracy: %.3f' % (accuracy * 100))





























