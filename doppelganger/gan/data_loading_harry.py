"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import pandas as pd
import sys
import os
import datetime
def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data





      
def sine_data_generation_f_a(no, seq_len, dim):
  # Initialize the output
  data = list()
  
  data_static_new = list()
  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
   
    temp_static_new = list()
    # For each feature
    
    for k in range(dim):
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
   

      temp_data_new = freq
      temp_static_new.append(temp_data_new)
      temp_data_new = phase
      temp_static_new.append(temp_data_new)


    # Align row/column
    temp = np.transpose(np.asarray(temp)) 
    temp_static_new = np.transpose(np.asarray(temp_static_new))

    # Stack the generated data
    data.append(temp)    
    data_static_new.append(temp_static_new)       
  return np.array(data), np.array(data_static_new)
      
      
def real_data_loading_prism():

  def dday(x):
    try:
        if data['id'][x.name] == data['id'][x.name - 1]:
            return (data['visit_date'][x.name] - data['visit_date'][x.name-1]).days
        else:
            return 0
    except KeyError:
        return 0

  
  data = pd.read_csv('./prism.csv')
  data = data[['Household_Id', 'Participant_Id', 'Visit date [EUPATH_0000091]', 'Weight (kg) [EUPATH_0000732]', 'Temperature (C) [EUPATH_0000110]', 'Age at visit (years) [EUPATH_0000113]', 'Height (cm) [EUPATH_0010075]']]
  data.columns = ['house_id', 'id', 'visit_date', 'weight', 'temp', 'age', 'height']
  
  data = data[data['visit_date'].isna() == False]
  data['visit_date'] = pd.to_datetime(data.visit_date)
  data.sort_values(['id', 'visit_date'], inplace=True)
  data.set_index(pd.Index(np.arange(len(data))), inplace=True)

  data['dday'] = data.apply(dday, axis=1)
  one_hot = pd.get_dummies(data['house_id'], drop_first=True)
  df = data.drop(['visit_date', 'house_id'], axis=1)
  df = df.join(one_hot)
  normalized_df=(df-df.min())/(df.max()-df.min())
  normalized_df['id'] = df['id']

  dimensions = ['dday', 'weight', 'height', 'age']
  max_ = []
  min_ = []
  for dim in dimensions:
    max_.append(df[dim].max())
    min_.append(df[dim].min())

  def format(data, min_, max_):
    data.interpolate(method = 'linear', inplace=True)
    #np.savez('./data/prism/data_train', data)
    dimensions = ['dday', 'weight', 'height', 'age']
    
    
    
    id_unique = data.id.unique()
    features = []
    attributes = []
    gen_flag = []
    for i in id_unique:
        child = np.array(data.loc[data['id'] == i][dimensions])
        if len(child) >= 5:
          gen_flag.append(np.concatenate([np.ones(len(child)), np.zeros(130-len(child))]))   
          child = np.pad(child, ((0, 130-len(child)), (0,0)))
          features.append(child)
          attributes.append(np.array(df.loc[df['id'] == i].iloc[:, 6:])[0])
          
          
    return np.array(features), np.array(attributes), np.array(gen_flag), min_, max_

  
    
  return format(normalized_df, min_, max_)




def renormalize(data, min_, max_):
  for child in data:
    new_child=[]
    for row in child:
      if np.any(row):
        
        new_row = np.multiply(row, np.subtract(max_, min_))#np.apply_along_axis(np.multiply, 0, row, np.subtract(max_, min_))

        new_row = np.add(new_row, min_)#np.apply_along_axis(np.add, 0, row, min_)
      else:
        new_row=row  
      new_child.append(np.array(new_row))
      
    new_child = np.expand_dims(new_child, axis=0)
    try:
      new_data = np.vstack([new_data, new_child])
    except UnboundLocalError:
      new_data = new_child
    
  
  return np.array(new_data)

def one_hot_encode(attributes):
  
  new_attributes = []
  for row in attributes:
    new_row = np.zeros_like(row)
    new_row[np.argmax(row)] = 1
    new_attributes.append(new_row)

  return(np.array(new_attributes))  
  

import pickle
from output import *
data_feature_output = [
    Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
    Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
    Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
    Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False)
]

with open('data/data_feature_output.pkl', 'wb') as f:
  pickle.dump(data_feature_output, f)

data_attribute_output = [Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False)]
with open('data/data_attribute_output.pkl', 'wb') as f:
    pickle.dump(data_attribute_output, f)


import tensorflow as tf
import numpy as np 
data_feature,_,data_gen_flag, min_, max_ = real_data_loading_prism()
data_attribute = np.ones(shape=(1347, 1))

print(data_feature.shape)
print(data_attribute.shape)
print(data_gen_flag.shape)
np.savez('data/data_train.npz', data_feature=data_feature, data_attribute=data_attribute, data_gen_flag=data_gen_flag)