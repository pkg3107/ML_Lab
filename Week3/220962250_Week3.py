#!/usr/bin/env python
# coding: utf-8

# # 1
# 
# 1. Consider the hepatitis/ pima-indians-diabetes csv file, perform the following date pre-processing.
# 1. Load data in Pandas.
# 2. Drop columns that aren’t useful.
# 3. Drop rows with missing values.
# 4. Create dummy variables.
# 5. Take care of missing data.
# 6. Convert the data frame to NumPy.
# 7. Divide the data set into training data and test data.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('/home/ustudent/Pranav_220962250/Week3/diabetes_csv.csv')


# In[3]:


df = df.drop(columns=['BloodPressure', 'SkinThickness'])
df


# In[4]:


df = df.dropna()
df


# In[5]:


df = df.fillna(df.median())
df


# In[6]:


arr = df.to_numpy()
arr


# In[9]:


X = df.drop(columns=['Outcome']).values
y = df['Outcome'].values

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# In[13]:


np.random.seed(42)

num_samples = X.shape[0]

indices = np.arange(num_samples)
np.random.shuffle(indices)

test_size = int(0.3 * num_samples)
train_size = num_samples - test_size

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

print("Training features:", X_train)
print("\nTesting features:", X_test)
print("\nTraining target:", y_train)
print("\nTesting target:", y_test)


# # 2. 
# a. Construct a CSV file with the following attributes:
# Study time in hours of ML lab course (x)
# Score out of 10 (y)
# The dataset should contain 10 rows.

# In[26]:


fd=pd.read_csv('/home/ustudent/Pranav_220962250/Week3/ml_lab_study_scores.csv')


# b. Create a regression model and display the following:
# 
# Coefficients: B0 (intercept) and B1 (slope)
# 
# RMSE (Root Mean Square Error)
# 
# Predicted responses

# In[31]:


X=fd['Study time in hours of ML lab course (x)'].values
Y=fd['Score out of 10 (y)']

n=len(X)

mean_x = np.mean(X)
mean_y = np.mean(Y)

num = np.sum((X-mean_x)*(Y-mean_y))
den = np.sum((X-mean_x)**2)

B1 = num/den
B0 = mean_y - B1 * mean_x

y_pred = B0 + B1 * X

rmse = np.sqrt(np.sum((Y - y_pred)**2)/n)

print(f'Coefficients: B0 {B0}, B1 {B1}')
print(f'RMSE: {rmse}')
print(f'Predicted y: \n{y_pred}')


# In[37]:


import matplotlib.pyplot as plt

plt.scatter(X, Y, color='red', label='Actual Data')
plt.plot(X, y_pred, color='blue', label='Predicted Data')

plt.xlabel('Study time in hours')
plt.ylabel('Score out of 10')
plt.legend()


# Additional Question

# In[40]:


df = pd.read_csv('/home/ustudent/Pranav_220962250/Week3/hepatitis_csv.csv')

X = df['bilirubin'].values
y = df['age'].values

n = len(X)
sum_x = X.sum()
sum_y = y.sum()
sum_x2 = (X ** 2).sum()
sum_xy = (X * y).sum()

B1_ped = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
B0_ped = (sum_y - B1_ped * sum_x) / n

y_pred_ped = B0_ped + B1_ped * X
rmse_ped = np.sqrt(((y - y_pred_ped) ** 2).mean())

A = np.array([[n, sum_x],
              [sum_x, sum_x2]])
b = np.array([sum_y, sum_xy])

B = np.linalg.solve(A, b)
B0_mat, B1_mat = B

y_pred_mat = B0_mat + B1_mat * X
rmse_mat = np.sqrt(((y - y_pred_mat) ** 2).mean())

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X, y_pred_ped, color='blue', label='Pedhazur Regression Line')
plt.xlabel('Bilirubin')
plt.ylabel('Age')
plt.title('Pedhazur Formula Method')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X, y_pred_mat, color='blue', label='Matrix Algebra Regression Line')
plt.xlabel('Bilirubin')
plt.ylabel('Age')
plt.title('Matrix Algebra Method')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Intercept (B0): {B0_ped:.2f}")
print(f"Slope (B1): {B1_ped:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse_ped:.2f}")

print(f"\nMatrix Algebra Method:")
print(f"Intercept (B0): {B0_mat:.2f}")
print(f"Slope (B1): {B1_mat:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse_mat:.2f}")

# Predicting value for


# In[ ]:




