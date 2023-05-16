
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd 
import seaborn as sns # Convention alias for Seaborn
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap    


#importing datasets 
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\Datasets\\CarManufactureUserData.csv")


#Extracting Independent and dependent Variable  
#x= df.iloc[:, [2,3]].values  
#y= df.iloc[:, 4].values  
#or use a readable format :
x =  df[['Age', 'EstimatedSalary']] .values  #dataset contains lots of information but the Estimated Salary and Age we will consider for the independent variable
y = df['Purchased'].values


# Splitting the dataset into training and test set.  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  


#feature Scaling 
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

#fitting K-NN classifier to the training set  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)

#predicting the test set result  
y_pred = classifier.predict(x_test)  

#creating the Confusion matrix
cm = confusion_matrix(y_test, y_pred) 

print(cm) 

#visulaizing the trianing set result  
x_set, y_set = x_train, y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('red','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
mtp.title('K-NN Algorithm (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()  























