
# The Iris dataset is one of the most well-known and frequently used datasets in machine learning and statistics. It was introduced by the British statistician and biologist Ronald Fisher in 1936 and has since become a standard benchmark for classification algorithms.
# The dataset consists of measurements from 150 iris flowers belonging to three different species: setosa, versicolor, and virginica. Each sample has four features or attributes:
# Sepal length (in centimeters): It represents the length of the sepal, which is the outer part that protects the flower bud.
# Sepal width (in centimeters): It denotes the width of the sepal.
# Petal length (in centimeters): It represents the length of the petal, which is the inner part of the flower that is usually more colorful.
# Petal width (in centimeters): It indicates the width of the petal.
# The goal of the Iris dataset is to classify the flowers into their respective species based on these four measurements. This is a classic supervised learning problem, where the dataset is labeled with the species of each sample.

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Iris.csv")

# extract features and target
X = df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

# fit and transform our target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into labeled and unlabeled subsets
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X,y,test_size=0.8,stratify=y)

# Create the Label Spreading model
model = LabelSpreading(kernel='knn', alpha=0.8)

# Fit the model using both labeled and unlabeled data
model.fit(X_labeled, y_labeled)

# Predict labels for the unlabeled data
y_pred = model.predict(X_unlabeled)

# Compute the accuracy of the model
accuracy = accuracy_score(y_unlabeled, y_pred)
print('Accuracy:', accuracy)