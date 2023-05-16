
# Use the Data.csv provided for which illustrates the ability of purchasing a product based on age, salary & country of origin.


import numpy as np 
#import matplotlib.pyplot as mpt 
import pandas as pd
import os  

#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute  import SimpleImputer  #pip install scikit-learn  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler   

data_set = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\Data.csv")

x= data_set.iloc[:,:-1].values #the first colon(:) is used to take all the rows, and the second colon(:) is for all the columns,used :-1, 
#because we don't want to take the last column as it contains the dependent variable 

y= data_set.iloc[:,3].values #taken all the rows with the last column only 
print(y)


#handling missing data (Replacing missing data with the mean value)
imputer= SimpleImputer(missing_values =np.nan, strategy='mean')
#Fitting imputer object to the independent variables x.   
imputer= imputer.fit(x[:, 1:3])  
#Replacing missing data with the calculated mean value  
x[:, 1:3]= imputer.transform(x[:, 1:3])     


#Catgorical data for Country Variable  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])   # encoded the variables into digits

X = x


#dummy variables are those variables which have values 0 or 1
#Encoding for dummy variables 
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(X)



#the purchased variable has only two categories yes or no, and which are automatically encoded into 0 and 1.
labelencoder_y= LabelEncoder()  
y= labelencoder_y.fit_transform(y)  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)  




#create the object of StandardScaler class for independent variables or features. And then we will fit and transform the training dataset
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  
# test dataset, we will directly apply transform() function instead of fit_transform() because it is already done in training set.
x_test= st_x.transform(x_test)  












print(x_train)

