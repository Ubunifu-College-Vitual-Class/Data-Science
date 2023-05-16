
#As  Data scientist you are required to provide information to the Counties regarding eligibility of adding more health facilities 
#i.e. you are required to come up with statistics of counties where there is an urgent need of addition of health a facility.
#Prepare the Dataset given for your training model.


import numpy as np 
#import matplotlib.pyplot as mpt 
import pandas as pd
import os  

#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute  import SimpleImputer  #pip install scikit-learn  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split  

data_set = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\DataExercise.csv")

x= data_set.iloc[:,:-1].values #the first colon(:) is used to take all the rows, and the second colon(:) is for all the columns,used :-1, 
#because we don't want to take the last column as it contains the dependent variable 


label_encoder_y = LabelEncoder()
y = data_set.iloc[:,4].values #taken all the rows with the last column only 

label_encoder_y= LabelEncoder()  
y = label_encoder_y.fit_transform(y) #nb is a 1 dimension array

print(y)


#encode you categorical column prone to malaria
label_encoder_x= LabelEncoder()  
x[:, 3]= label_encoder_x.fit_transform(x[:, 3])

#handling missing data (Replacing missing data with the mean value)
imputer = SimpleImputer(missing_values =np.nan, strategy='mean')
#Fitting imputer object to the independent variables x.   
imputer= imputer.fit(x[:, 1:2])  
#Replacing missing data with the calculated mean value  
x[:, 1:2]= imputer.transform(x[:, 1:2])   



#fill in the nana in the 3rd index
imputer= imputer.fit(x[:, 1:3])  
#Replacing missing data with the calculated mean value  
x[:, 1:3]= imputer.transform(x[:, 1:3])   
   

#encode you categorical column on county on x
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])

#split your training and test data
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  

print(x_train)