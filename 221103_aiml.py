#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


df=pd.read_csv('bottle.csv')


# In[ ]:


df_binary=df[['Salnty','T_degC']]


# In[ ]:


#Taking only the selected two attributes from the dataset
df_binary.columns=['Sal','Temp']


# In[ ]:


#Renaming the columns from easier wrirting of the code
df_binary.head()


# In[ ]:


#Plotting the data scatter
sns.lmplot(x="Sal",y="Temp",data=df_binary,order=2,ci=None)


# In[ ]:


#Data cleaning
#Eliminating NaN or missing input numbers
df_binary.fillna(method='ffill',inplace=True)


# In[ ]:


X=np.array(df_binary['Sal']).reshape(-1,1)
y=np.array(df_binary['Temp']).reshape(-1,1)


# In[ ]:


#Seperating the data into independant and dependant variables
#Converting each dataframe into a numpy array(since each dataframe contains only one column)
df_binary.dropna(inplace=True)


# In[ ]:


#Dropping anuy rows with Nan values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[ ]:


#Splitting the data into training and testing data
regr=LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_test,y_test))


# In[ ]:


y_pred=regr.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[ ]:


df_binary500=df_binary[:][:500]


# In[ ]:


#Selecting the 1st 500 rows of the data
sns.lmplot(x="Sal",y="Temp",data=df_binary500,order=2,ci=None)


# In[ ]:


df_binary500.fillna(method='ffill',inplace=True)


# In[ ]:


X=np.array(df_binary500['Sal']).reshape(-1,1)
y=np.array(df_binary500['Temp']).reshape(-1,1)


# In[ ]:


df_binary500.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# In[ ]:


regr=LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_test,y_test))


# In[ ]:


y_pred=regr.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




