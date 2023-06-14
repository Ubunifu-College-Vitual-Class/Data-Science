# Build a semi-supervised learning model to classify news articles into different categories using a limited number of labelled examples and a large number of unlabelled examples.
# The dataset is attached that contains new articles along with their corresponding categories.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import os

#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\News Articles Classification Extracted.csv")

#encode categorical column
cols = ['Title', 'Description']
le = LabelEncoder(); 
df[cols] = df[cols].apply(le.fit_transform)

# extract features and target
X_labeled = df[['Title', 'Description']]
y_labeled = df['Categories']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=0, stratify=y_labeled) # stratify=y argument ensures that the class distribution in y is preserved in both the training and testing sets

# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

# summarize training set size
print('Labeled Train Set:', X_train_lab.shape, y_train_lab.shape)
print('Unlabeled Train Set:', X_test_unlab.shape, y_test_unlab.shape)
# summarize test set size
print('Test Set:', X_test.shape, y_test.shape)

# define model
model = LogisticRegression()
# fit model on labeled dataset
model.fit(X_train_lab, y_train_lab)


# make predictions on hold out test set
yhat = model.predict(X_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))







