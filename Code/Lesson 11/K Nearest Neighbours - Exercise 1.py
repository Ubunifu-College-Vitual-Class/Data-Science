
# Using the Diabetes Dataset link :
#  https://www.kaggle.com/datasets/mathchi/diabetes-data-set/download?datasetVersionNumber=1 
# Females of at least 21years , health records that includes the number of pregnancies , glucose , blood pressure , skin thickness , insulin , BMI , Diabetes Pedigree function and Age were recorded. 
# They were later tested for diabetes and results recorded as the Outcome.
# Create a model to predict the outcome of a female with the above records
# Display the classification report for your model

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # Convention alias for Seaborn
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
 


#importing datasets 
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Diabetes.csv")

# scatter plot matrix
# sns.pairplot(df, hue='Outcome')
# plt.show()


x =  df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] .values  #dataset contains lots of information but the Estimated Salary and Age we will consider for the independent variable
y = df['Outcome'].values


# Splitting the dataset into training and test set.  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  


#feature Scaling 
st_x= StandardScaler() #data normalization , performed during the data preprocessing   
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

#fitting K-NN classifier to the training set  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)

#predicting the test set result  
y_pred = classifier.predict(x_test)  

#creating the Confusion matrix
cm = confusion_matrix(y_test, y_pred) 

#to see how accurate our algorithm is
print(cm)
print(classification_report(y_test, y_pred))
#accuracy - how much percentage of cases that you have predicted correctly
#recall - how much percentage of real positive cases are correctly identified
#precision - among the cases predicted to be positive, how much percentage of them are really positive
#f1-score - harmonic mean of recall and precision
#harmonic mean is a type of average that is calculated by dividing the number of values in a data series by the sum of the reciprocals

accuracy = accuracy_score(y_test, y_pred)#measure of how well the classifier predicts the correct class labels for new data points.
#It is calculated as the ratio of the number of correctly predicted data points to the total number of data points.
print(f'K-NN model accuracy : {accuracy*100:.0f} %')

# Over 90% - Very good.
# Between 70% and 90% - Good.
# Between 60% and 70% - OK.
# Below 60% - Poor.


#pregnancy #glucose # bloodpressure #skinthickness #insulin #bmi #diabetespedigreefunction #age
predictDiabetic = [1,150,72,32,200,35,0.191,28]
diabetic_risk = classifier.predict([predictDiabetic])

print("++++++++++++++++++++++++++++++++++++++")
#Diabetic 1 = high 0 = low
print(f'Diabetic outcome of a female with the above records is {diabetic_risk[0]:.0f}')
print("++++++++++++++++++++++++++++++++++++++")























