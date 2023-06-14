# evaluate logistic regression fit on label propagation for semi-supervised learning
from numpy import concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LogisticRegression
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y) # stratify=y argument ensures that the class distribution in y is preserved in both the training and testing sets

# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab)) # join

# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]

# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))

# define model
model = LabelPropagation()

# fit model on training dataset
model.fit(X_train_mixed, y_train_mixed)

# get labels for entire training dataset data
tran_labels = model.transduction_

# define supervised learning model
model2 = LogisticRegression()

# fit supervised learning model on entire training dataset
model2.fit(X_train_mixed, tran_labels)

# make predictions on hold out test set
yhat = model2.predict(X_test)

# calculate score for test set
score = accuracy_score(y_test, yhat)

# summarize score
print('Accuracy: %.3f' % (score*100))