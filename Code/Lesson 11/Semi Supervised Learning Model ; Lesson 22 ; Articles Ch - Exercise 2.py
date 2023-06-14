# evaluate label propagation on the semi-supervised learning dataset
from numpy import concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd


#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\News Articles Classification Extracted.csv")

#encode categorical column
cols = ['Title']
le = LabelEncoder(); 
df[cols] = df[cols].apply(le.fit_transform)


# extract features and target
X = df[['Title']].values
y = df['Categories'].values


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y) # stratify -  data splitting process will ensure that the training and testing sets have similar class distributions as the original dataset


# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)


# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab)) # combine multiple arrays or data structures into a single array


# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]
# recombine training dataset labels


y_train_mixed = concatenate((y_train_lab, nolabel))


# define model
model = LabelPropagation()

# fit model on training dataset
model.fit(X_train_mixed, y_train_mixed)
# make predictions on hold out test set
yhat = model.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))