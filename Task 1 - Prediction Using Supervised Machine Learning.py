#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation : GRIP 2022
# 
# ### Author : Aniket Kiran Adhav 
# 
# ### Data Science and Business Analytics Intern 
# 
# ### Task1 : Prediction Using Supervised ML 
# 

# # Importing the dataset

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 

# To ignore the warnings 
import warnings as wg
wg.filterwarnings("ignore")


# # Reading Data From Remote Link

# In[2]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# # Plotting the distribution of scores

# In[3]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


#  ### **Preparing the data**
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[4]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# ### Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


#  ### **Training the Algorithm**
# We have split our data into training and testing sets, and now is finally the time to train our algorithm. 

# In[6]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# # Plotting the regression line

#  line = regressor.coef_*X+regressor.intercept_
# 
#  Plotting for the test data
# plt.scatter(X, y)
# plt.plot(X, line);
# plt.show()

# # **Making Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# In[7]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# ### Comparing Actual vs Predicted

# In[8]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# ## You can also test with your own data

# In[ ]:


hours = float(input('Enter the number of hours the student studies: '))
prediction = model.predict([[hours]])

print('The nuber of hours {}'.format(hours));print('Percentage based om it is: {}'.format(prediction[0]))
print('Prediction is sucessful')


# No of Hours = 9.25
# Predicted Score = 93.69173248737539

#  ### **Evaluating the model**
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[ ]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:


Mean Absolute Error: 4.183859899002982

