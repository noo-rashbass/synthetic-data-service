#!/usr/bin/env python
# coding: utf-8

# Construct a Data Frame, <font color = blue> Observation_df </font>, which sorts each column by different values(aka categories) in each column, and how many data are there in each category.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
observations = pd.read_csv('https://raw.githubusercontent.com/noo-rashbass/synthetic-data-service/master/Aisha/observations.csv')
observations['count']=pd.Series(np.ones(observations.shape[0],dtype=int)) #Add a counter
observ_columns = (observations.keys()).tolist()
observ_columns = observ_columns[:-1] #An array storing column names, discarding the added 'count' column


# In[2]:


data_volumn = observations.shape[0]
avail_rate = pd.Series(np.zeros(observations.shape[1]-1),index=observ_columns)
no_cate_series = pd.Series(np.zeros(observations.shape[1]-1,dtype=int),index=observ_columns)
index_array = []
layer_array = []
value_list = []
for column in observ_columns:
    p=observations.groupby([column])['count'].sum()
    rate = np.sum(p.values) * 100 / data_volumn
    no_cate = p.shape[0]
    avail_rate[column]=rate
    no_cate_series[column]=no_cate
    
    if (len(p)== data_volumn):
        p = pd.Series(1, index=['unique for each'])
    elif (len(p)==1):
        p = pd.Series(data_volumn, index=['same for all'])
    
    index_array=index_array+[column]*len(p)
    p_layer=(p.index).tolist()
    layer_array = layer_array + p_layer
    value_list = value_list+list(p.values)

multilayer_array=[index_array,layer_array]
index_fin = pd.MultiIndex.from_arrays(multilayer_array, names=('Column', 'Category'))
Observation_df = pd.DataFrame({'Count': value_list},index=index_fin)


# By here, the <font color=blue>Observation_df</font> has been constructed. The next code displays the Observation_df dataframe.

# In[3]:


for name in observ_columns[:-1]:
    display(Observation_df.loc[[name]])


# In the end, there is a dataframe showing the proportion of "non-NA" values, and how many categories in each column.

# In[4]:


avail_rate_df = pd.DataFrame({'non-NA proportion in %':avail_rate,'no of categories':no_cate_series})
display(avail_rate_df)


# In[ ]:




