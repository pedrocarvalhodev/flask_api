
# coding: utf-8

# In[1]:


import os 
import json
import numpy as np
import pandas as pd


# In[2]:


path="/home/pedro/repos/ml_web_api/flask_api"


# In[3]:


data = pd.read_csv(path+'/data/training.csv')


# In[4]:


data.head(2)


# In[5]:


for _ in data.columns:
    print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))

