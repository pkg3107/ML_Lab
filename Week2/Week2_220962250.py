#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# 1. Follow along with these steps:
# 
# a) Create a figure object called fig using plt.figure()
# 
# b) Use add_axes to add an axis to the figure canvas at [0,0,1,1]. Call this new axis ax.
# 
# c) Plot (x,y) on that axes and set the labels and titles to match the plot below:

# In[2]:


fig=plt.figure()
ax=fig.add_axes([0, 0, 1, 1])

x=[0,100]
y=[0,200]

ax.plot(x, y, label='y=2x')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Title')


# 2. Create a figure object and put two axes on it, ax1 and ax2. Located at [0,0,1,1] and [0.2,0.5,.2,.2]
# respectively. Now plot (x,y) on both axes. And call your figure object to show it.

# In[3]:


fig2 = plt.figure(figsize=(10,6))

ax1 = fig2.add_axes([0, 0, 1, 1])
ax2 = fig2.add_axes([0.2, 0.5, 0.2, 0.2])

x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]

ax1.plot(x, y)
ax2.plot(x, y)


# 3. Use the company sales dataset csv file, read Total profit of all months and show it using a line plot
# Total profit data provided for each month. Generated line plot must include the following properties: –
# 
# a. X label name = Month Number
# 
# b. Y label name = Total profit

# In[4]:


df=pd.read_csv('/home/ustudent/Pranav_220962250/Week2/company_sales_data.csv')

months = df['month_number']
total_profit = df['total_profit']

plt.figure()
plt.plot(months, total_profit)

plt.title('Month-wise Profit')
plt.xlabel('Month Number')
plt.ylabel('Total Profit')


# 4. Use the company sales dataset csv file, get total profit of all months and show line plot with the following
# Style properties. Generated line plot must include following Style properties: –
# 
# a. Line Style dotted and Line-color should be red
# 
# b. Show legend at the lower right location.
# 
# c. X label name = Month Number
# 
# d. Y label name = Sold units number
# 
# e. Add a circle marker.
# 
# f. Line marker color as read
# 
# g. Line width should be 3

# In[5]:


months=df['month_number']
units=df['total_units']

plt.figure()
plt.plot(months, units, linestyle=':', color='red', marker='o', linewidth=3, markerfacecolor='red', label='Total Units')

plt.xlabel('Month Number')
plt.ylabel('Sold units number')

plt.legend(loc='lower right')

plt.xticks(months)


# Additional Questions
# 

# 1. Use the company sales dataset csv file, read all product sales data and show it using a multiline plot.
# Display the number of units sold per month or each product using multiline plots. (i.e., Separate Plotline
# for each product ).

# In[6]:


months=df['month_number']
products=['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer']

plt.figure()

for i in products:
    plt.plot(months, df[i], marker='o')

plt.xlabel('No. of items')
plt.ylabel('Month number')
plt.legend(products)

plt.title('Monthwise sale of each product')


# 2. Use the company sales dataset csv file, calculate total sale data for last year for each product and show it
# using a Pie chart.

# In[8]:


products=['facecream', 'facewash', 'toothpaste', 'bathingsoap', 'shampoo', 'moisturizer']

tot_sales=[df['facecream'].sum(), df['facewash'].sum(), df['toothpaste'].sum(), 
           df['bathingsoap'].sum(), df['shampoo'].sum(), df['moisturizer'].sum()]

plt.figure()
plt.pie(tot_sales, labels=products, autopct='%1.1f%%', startangle=0)
plt.axis('equal')
plt.title('Product wise share of sale')


# In[ ]:




