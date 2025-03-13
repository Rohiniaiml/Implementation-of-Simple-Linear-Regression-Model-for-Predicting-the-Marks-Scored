# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ROHINI R
RegisterNumber: 212224240138 
*/
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('student_scores.csv')

print(df)

df.head(0)

df.tail(0)

print(df.head())

print(df.tail())

x = df.iloc[:,:-1].values

print(x)

y = df.iloc[:,1].values

print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

mae = mean_absolute_error(y_test,y_pred)

print("MAE: ",mae)

mse = mean_squared_error(y_test,y_pred)

print("MSE: ",mse)

rmse = np.sqrt(mse)

print("RMSE: ",rmse)

graph plotting for trainig set data
plt.scatter(x_train,y_train)

plt.plot(x_train,regressor.predict(x_train) , color ='blue')

plt.title("Hours vs Scores(training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

graph plotting for test set data
plt.scatter(x_test,y_test)

plt.plot(x_test,regressor.predict(x_test),color = 'black')

plt.title("Hours vs Scores(testing set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()
```

## Output:
![Screenshot 2025-03-13 082408](https://github.com/user-attachments/assets/55bb3eb1-7c5d-40f0-a081-68b55acc8372)
![Screenshot 2025-03-13 082417](https://github.com/user-attachments/assets/bf85b9af-c13b-45e2-9e5f-3759fe1e84bb)

![Screenshot 2025-03-13 082534](https://github.com/user-attachments/assets/21cc724b-9f31-4908-a2c9-80ac2e58587e)
![Screenshot 2025-03-13 082523](https://github.com/user-attachments/assets/c1aaf1fa-4d76-4942-aa8e-6b467808022c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
