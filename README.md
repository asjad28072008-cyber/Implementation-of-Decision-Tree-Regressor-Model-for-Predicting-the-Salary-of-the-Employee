# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Muhammad Asjad E
RegisterNumber:  25013957
*/


import pandas as pd 
data=pd.read_csv("Salary.csv") 
data.head() 
data.info() 
data.isnull().sum() 
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder() 
data["Position"]=le.fit_transform(data["Position"]) 
data.head() 
x=data[["Position","Level"]] 
x.head() 
y=data["Salary"] 
y.head() 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2) 
from sklearn.tree import DecisionTreeRegressor 
dt=DecisionTreeRegressor() 
dt.fit(x_train,y_train) 
y_pred=dt.predict(x_test) 
print(y_pred )

mse=metrics.mean_squared_error(y_test, y_pred)
print(mse)
r2= metrics.r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
```

## Output:
Data Head:

<img width="581" height="361" alt="Screenshot 2025-10-06 143116" src="https://github.com/user-attachments/assets/0af04a54-8625-4e44-af4b-241d5614a46a" />


isnull().sum()


<img width="299" height="324" alt="Screenshot 2025-10-06 143208" src="https://github.com/user-attachments/assets/8560efef-367f-4f8e-8f85-c73cb7cd4c0c" />



Data Head for salary:


<img width="503" height="374" alt="Screenshot 2025-10-06 143438" src="https://github.com/user-attachments/assets/b3715739-c6ab-4912-bfa7-f4fe1a02da47" />



Data info:



<img width="460" height="249" alt="Screenshot 2025-10-06 143025" src="https://github.com/user-attachments/assets/f086de99-dec4-461e-8e3e-27061eb8b7c7" />




Mean Squared Error and R2 Value:



<img width="345" height="46" alt="Screenshot 2025-10-06 142830" src="https://github.com/user-attachments/assets/d66a58c5-114c-499f-b702-e4b031c14f74" />



Data Prediction:


<img width="260" height="38" alt="Screenshot 2025-10-06 142933" src="https://github.com/user-attachments/assets/58ac6488-7484-487c-97a8-3d7d32ffd8c2" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
