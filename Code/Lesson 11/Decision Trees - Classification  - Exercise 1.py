import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree # plot the tree



#read our CSV data file
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\KCBankUserData.csv")

# Encode labels of multiple columns at once
cols = ['Risk', 'History', 'Debt', 'Collateral', 'Income']
cols_en = ['Risk_Encoded', 'History_Encoded', 'Debt_Encoded', 'Collateral_Encoded', 'Income_Encoded']
df[cols_en] = df[cols].apply(LabelEncoder().fit_transform)
#print(df.head())


#target variable = Risk
y = df['Risk_Encoded']
#features 
x = df.drop(['Risk_Encoded', 'Risk', 'History', 'Debt', 'Collateral', 'Income'], axis=1)

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

#create the object of StandardScaler class for independent variables or features. And then we will fit and transform the training dataset
st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train)  
# test dataset
X_test= st_x.transform(X_test)  


#train the algorithm
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#make predictions on the test data
y_pred = classifier.predict(X_test)

#to see how accurate our algorithm is
#print(confusion_matrix(y_test, y_pred))
#accuracy - how much percentage of cases that you have predicted correctly
#recall - how much percentage of real positive cases are correctly identified
#precision - among the cases predicted to be positive, how much percentage of them are really positive
#f1-score - harmonic mean of recall and precision
#harmonic mean is a type of average that is calculated by dividing the number of values in a data series by the sum of the reciprocals 
print(classification_report(y_test, y_pred))

#history #Debt # Collateral #Income
predictRisk = [1,0,1,2]
credit_risk = classifier.predict([predictRisk])



print("++++++++++++++++++++++++++++++++++++++")
print("Customer earning over $35, with no collateral and having a high debt in addition of a good history has credit risk :")
#credit 0 = high 1 = low
print(credit_risk)
print("++++++++++++++++++++++++++++++++++++++")

tree.plot_tree(classifier);