#!/usr/bin/env python
# coding: utf-8

# In[100]:


import warnings
warnings.simplefilter('ignore')


# In[101]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[102]:


df = pd.read_csv('hiring.csv')
df.head()


# In[103]:


df.isna().sum()


# In[104]:


df['experience'].fillna(0, inplace = True)


# In[108]:


df.info()


# In[109]:


df['test_score'].fillna(df['test_score'].mean(), inplace=True)


# In[110]:


df.isna().sum()


# In[111]:


X = df.iloc[:,:-1]
y = df['salary']


# In[112]:


def convert(x):
    dict = {'two':2,'three':3,'five':5,'seven':7,'ten':10,'eleven':11, 0:0}
    return dict[x]


# In[113]:


X['experience'] = X['experience'].apply(lambda x: convert(x))


# In[114]:


X.head()


# In[115]:


from sklearn.linear_model import LinearRegression


# In[116]:


reg = LinearRegression()


# In[117]:


reg.fit(X,y)


# In[120]:


ypred = reg.predict(X)


# In[121]:


y


# In[123]:


from sklearn.metrics import r2_score
r2_score(ypred,y)


# In[131]:


reg.predict([[1,0,1]])


# In[134]:


pickle.dump(reg, open('model.pkl','wb'))


# In[135]:


model2 = pickle.load(open('model.pkl','rb'))
model2.predict([[14,19,10]])


# In[ ]:




