import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Convention alias for Seaborn
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report



#Loading Data
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Diabetes.csv")

#create new df_col dataframe from df.copy() method.
df_col = df.copy()
#rename columns name
df_col.rename(columns={"Pregnancies" : "pregnant" , "Glucose" : "glucose" , "BloodPressure": "bp","SkinThickness": "skin","Insulin": "insulin","BMI": "bmi","DiabetesPedigreeFunction": "pedigree","Age": "age","Outcome": "label"} , inplace = True)
# renamed_df = df_col.head(3)
df = df_col

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
x = df[feature_cols] # Features
y = df.label # Target variable


#split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)
# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

#evaluate the performance of a classification model
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#visualize the confusion matrix using Heatmap
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#evaluate the model using classification_report for accuracy, precision, and recall
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))

