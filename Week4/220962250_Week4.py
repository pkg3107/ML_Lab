#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### 1) Yield vs Temp

# In[126]:


df=pd.read_csv('/home/ustudent/Pranav_220962250/Week4/experiment.csv')


# In[127]:


X=df['temp'].values
Y=df['yield'].values
n=15


# #### Simple Linear
# Pedhazur formula

# In[128]:


mean_x = np.mean(X)
mean_y = np.mean(Y)

num = np.sum((X-mean_x)*(Y-mean_y))
den = np.sum((X-mean_x)**2)

B1 = num/den
B0 = mean_y - B1 * mean_x

y_pred = B0 + B1 * X


# In[129]:


mse = (np.sum((Y - y_pred)**2)/n)
rmse = np.sqrt(mse)
print(f'Simple linear equation: y={B0:.5f}+ {B1:.5f}*x')
print(f'MSE: {mse}, RMSE: {rmse}')


# Calculus method

# In[130]:


X_design = np.vstack([np.ones(n), X]).T

coefficients = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y
B0_calculus, B1_calculus = coefficients

print(f'Calculus Method - B0 (intercept): {B0_calculus}')
print(f'Calculus Method - B1 (slope): {B1_calculus}')


# #### Polynomial Regression

# In[131]:


mat_X = np.array([
    [np.sum(X**4), np.sum(X**3), np.sum(X**2)],
    [np.sum(X**3), np.sum(X**2), np.sum(X)],
    [np.sum(X**2), np.sum(X), len(X)]
])

mat_Y = np.array([
    [np.sum(Y * (X**2))],
    [np.sum(Y * X)],
    [np.sum(Y)]
])

B2, B1, B0 = np.linalg.solve(mat_X, mat_Y)

print(f'B0 = {B0[0]:.5f} , B1 = {B1[0]:.5f} , B2 = {B2[0]:.5f}')

y_poly = B0 + B1 * X + B2 * (X**2)


# In[132]:


mse = (np.sum((Y - y_poly)**2)/n)
rmse = np.sqrt(mse)

print(f'Polynomial equation: y={B0} + {B1}*x + {B2}*x**2')
print(f'MSE: {mse}, RMSE: {rmse}')


# In[133]:


plt.scatter(X, Y, color='red')
plt.plot(X, y_pred, color='blue')
plt.plot(X, y_poly, color='green')
plt.title('Scatter plot of actual vs predicted values')
plt.xlabel('Temperature')
plt.ylabel('Yield')


# ### 2) Heart infection

# In[134]:


fd=pd.read_csv('/home/ustudent/Pranav_220962250/Week4/heart.csv')


# In[135]:


X1 = fd['Area']
X2 = fd['X2']
y = fd['Infarc']
X3 = fd['X3']


# In[136]:


mat1 = np.array([[len(X1),np.sum(X1),np.sum(X2),np.sum(X3)],
                 [np.sum(X1),np.sum(X1**2),np.sum(X1*X2),np.sum(X1*X3)],
                 [np.sum(X2),np.sum(X1*X2),np.sum(X2**2),np.sum(X3*X2)],
                 [np.sum(X3),np.sum(X1*X3),np.sum(X2*X3),np.sum(X2)]])

mat2 = np.array([[np.sum(y),np.sum(X1*y),np.sum(X2*y),np.sum(X3*y)]])

coeffs = np.dot(np.linalg.inv(mat1),mat2.T)

b0,b1,b2,b3 = coeffs[0,0],coeffs[1,0],coeffs[2,0],coeffs[3,0]

y_prd = b0 + b1*X1+ b2*(X2) +b3*X3
squared_err = (y-y_prd)**2

rmse_mat = np.sqrt(np.mean(squared_err))

print("Mean Squared Error: ")
print(np.mean(squared_err))

print("Coefficients: ")
print(f"b0: {b0}\nb1: {b1}\nb2: {b2}\nb3: {b3}")

print("y_pred: ")
print(y_prd)

print("RMSE matrix:")
print(rmse_mat)


# In[ ]:





# In[ ]:




