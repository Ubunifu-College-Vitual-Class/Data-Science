import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# You are developing a model to classify games at which machine learning will beat the world champion within five years.
# The following table contains the data you have collected.

# Build the optimal decision tree using the table above
# Display the classification report and accuracy score

# importing the dataset  
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\ML_Games.csv")

# Encode labels of multiple columns at once
cols = ['x1', 'x2', 'x3', 'y']
df = df[cols].apply(LabelEncoder().fit_transform)

print(df)

#features 
x = df.drop(['y'], axis=1)
#target variable = y(Win or Lose)
y = df['y']


#split the data into training and testing sets
#, 80/20% to 90/10% for large datasets;small dimensional datasets,  60/40% to 70/30%.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 0)

#train the algorithm
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#make predictions on the test data
y_pred = classifier.predict(X_test)

# Display the classification report and accuracy score
print(classification_report(y_test, y_pred))


accuracy = accuracy_score(y_test, y_pred)#measure of how well the classifier predicts the correct class labels for new data points.
#It is calculated as the ratio of the number of correctly predicted data points to the total number of data points.
print(f'K-NN model accuracy : {accuracy*100:.0f} %')























