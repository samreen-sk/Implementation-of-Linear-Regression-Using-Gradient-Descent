# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters – Set initial values for slope m and intercept 𝑏 and choose a learning rate 𝛼
2. Compute Cost Function – Calculate the Mean Squared Error (MSE) to measure model performance.
3. Update Parameters Using Gradient Descent – Compute gradients and update m and b using the learning rate.
4. Repeat Until Convergence – Iterate until the cost function stabilizes or a maximum number of iterations is reached.
## Program and Output:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SHAIK SAMREEN
RegisterNumber: 212223110047 
*/
```
```
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)

        theta -=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()
```
![423051849-f9e38d81-c7b2-470a-9bee-108c1e1dfbd6](https://github.com/user-attachments/assets/d99f679c-87f4-4893-9daf-a73cf316fc35)
```
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
```
![423051952-21db5ae7-0394-485b-83d5-6474b5dfc9d4](https://github.com/user-attachments/assets/50457cee-2581-414e-8518-8685f03470a6)
```
print(X1_Scaled)
```
![423052029-d7a63915-5a3e-4ed0-a78a-f199f9e11888](https://github.com/user-attachments/assets/ed3199ad-04f4-4b43-8d64-91ec71b7e7a3)

```
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
![423052154-a3f86c69-3c16-421e-bec6-379955c5b7ef](https://github.com/user-attachments/assets/bf25c213-34d5-4a6f-bc15-56534f424be0)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
