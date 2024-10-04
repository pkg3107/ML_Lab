#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[27]:


data  = pd.DataFrame({'Outlook' : ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
                     'Temp':[85,80,83,70,68,65,64,72,69,75,75,72,81,71],
                     'Humidity':[85,90,78,96,80,70,65,95,70,80,70,90,75,80],
                     'Wind':['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
                     'Decision':['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']})


# In[28]:


data


# In[39]:


#1
def entropy(y):
    if len(y) == 0:
        return 0
    probs = np.array([(np.sum(y==c)/len(y)) for c in np.unique(y)])
    return -np.sum(probs * np.log2(probs + 1e-9))

def information_gain(data,feature, target):
    tot_entrp = entropy(data[target])
    values = data[feature].unique()
    weighted_entrp = 0
    
    for value in values:
        subset = data[data[feature] == value]
        weighted_entrp += (len(subset)/len(data)) * entropy(subset[target])
    
    return tot_entrp - weighted_entrp

def build_c45(data,target,features):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    if len(features) == 0:
        return data[target].mode()[0]
    gains = [information_gain(data,feature,target) for feature in features]
    best_feature = features[np.argmax(gains)]
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = build_c45(subset,target,features[features != best_feature])
        tree[best_feature][value] = subtree
    
    return tree

def c45_classify(tree,sample):
    if not isinstance(tree,dict):
        return tree
    feature = next(iter(tree))
    value = sample[feature]
    if value in tree[feature]:
        return c45_classify(tree[feature][value], sample)
    else:
        return None


# In[46]:


features = data.columns[:-1]
c45_tree = build_c45(data, 'Decision', features)
new_sample = {'Outlook': 'Sunny', 'Temp': 80, 'Humidity': 70, 'Wind': 'Weak'}
c45_prediction = c45_classify(c45_tree, new_sample)
print(f"C4.5 Prediction for {new_sample}: {c45_prediction}")
print(c45_tree)


# In[31]:


#2
def gini_impurity(y):
    if len(y) == 0:
        return 0
    probs = np.array([(np.sum(y == c) / len(y)) for c in np.unique(y)])
    return 1 - np.sum(probs ** 2)

def best_categorical_split(data, feature, target):
    unique_values = data[feature].unique()
    best_gini = float('inf')
    best_split_value = None
    
    for value in unique_values:
        left_split = data[data[feature] == value]
        right_split = data[data[feature] != value]
        
        if len(left_split) == 0 or len(right_split) == 0:
            continue
        
        weighted_gini = (len(left_split) / len(data)) * gini_impurity(left_split[target]) + \
                        (len(right_split) / len(data)) * gini_impurity(right_split[target])
        
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_split_value = value
            
    return best_split_value

def best_numerical_split(data, feature, target):
    unique_values = np.sort(data[feature].unique())
    best_gini = float('inf')
    best_split_value = None
    
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2  # Midpoint
        left_split = data[data[feature] <= threshold]
        right_split = data[data[feature] > threshold]
        
        if len(left_split) == 0 or len(right_split) == 0:
            continue
        
        weighted_gini = (len(left_split) / len(data)) * gini_impurity(left_split[target]) + \
                        (len(right_split) / len(data)) * gini_impurity(right_split[target])
        
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_split_value = threshold
            
    return best_split_value


def build_cart_tree(data, target, features):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    
    if len(features) == 0:
        return data[target].mode()[0]
    
    best_feature = None
    best_split_value = None
    best_gini = float('inf')
    
    for feature in features:
        if data[feature].dtype == 'object':  
            split_value = best_categorical_split(data, feature, target)
        else:  
            split_value = best_numerical_split(data, feature, target)

        if split_value is not None:
            left_split = data[data[feature] == split_value] if data[feature].dtype == 'object' else data[data[feature] <= split_value]
            right_split = data[data[feature] != split_value] if data[feature].dtype == 'object' else data[data[feature] > split_value]
            
            if len(left_split) == 0 or len(right_split) == 0:
                continue
            
            weighted_gini = (len(left_split) / len(data)) * gini_impurity(left_split[target]) + \
                            (len(right_split) / len(data)) * gini_impurity(right_split[target])
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_split_value = split_value
    
    if best_feature is None:
        return data[target].mode()[0]  
    
    tree = {best_feature: {'<= ' + str(best_split_value): {}, '> ' + str(best_split_value): {}}}
    
    left_split = data[data[best_feature] <= best_split_value] if data[best_feature].dtype != 'object' else data[data[best_feature] == best_split_value]
    right_split = data[data[best_feature] > best_split_value] if data[best_feature].dtype != 'object' else data[data[best_feature] != best_split_value]
    
    tree[best_feature]['<= ' + str(best_split_value)] = build_cart_tree(left_split, target, features[features != best_feature])
    tree[best_feature]['> ' + str(best_split_value)] = build_cart_tree(right_split, target, features[features != best_feature])
    
    return tree

def cart_classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    split_value = float(next(iter(tree[feature])).split()[1]) if '>' in next(iter(tree[feature])) else next(iter(tree[feature])).split()[1]
    
    if isinstance(split_value, str):
        if sample[feature] == split_value:
            return cart_classify(tree[feature]['<= ' + split_value], sample)
        else:
            return cart_classify(tree[feature]['> ' + split_value], sample)
    else:
        if sample[feature] <= split_value:
            return cart_classify(tree[feature]['<= ' + str(split_value)], sample)
        else:
            return cart_classify(tree[feature]['> ' + str(split_value)], sample)


# In[47]:


cart_tree = build_cart_tree(data, 'Decision', features)
cart_prediction = cart_classify(cart_tree, new_sample)
print(f"CART Prediction for {new_sample}: {cart_prediction}")
print(cart_tree)


# In[37]:


#3
data2 = pd.DataFrame({'Income':['Low','Low','Medium','Medium','High','High'],
                     'Credit':['Good','Bad','Good','Bad','Good','Bad'],
                     'LoanAppr':['Yes','No','Yes','Yes','Yes','No']})


# In[49]:


features1 = data2.columns[:-1]
c45_tree2 = build_c45(data2, 'LoanAppr', features1)
new_sample = {'Income': 'Low', 'Credit': 'Bad'}
c45_prediction = c45_classify(c45_tree2, new_sample)
print(f"C4.5 Prediction for {new_sample}: {c45_prediction}")
print(c45_tree)


# In[50]:


cart_tree2 = build_cart_tree(data2, 'LoanAppr', features1)
cart_prediction = cart_classify(cart_tree2, new_sample)
print(f"CART Prediction for {new_sample}: {cart_prediction}")
print(cart_tree)


# In[51]:


def print_tree(tree, depth=0):
    indent = "    " * depth  # Indentation for depth
    if not isinstance(tree, dict):
        print(f"{indent}--> {tree}")  # Leaf node
        return
    
    for key, value in tree.items():
        print(f"{indent}{key}")  # Print the current node
        print_tree(value, depth + 1)


# In[56]:


print_tree(c45_tree)
print('---------------------------------------')
print_tree(cart_tree)
print('---------------------------------------')
print_tree(c45_tree2)
print('---------------------------------------')
print_tree(cart_tree2)
print('---------------------------------------')


# In[ ]:




