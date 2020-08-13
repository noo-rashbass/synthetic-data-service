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
import os
import sys
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


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data


def real_data_loading_prism():

  def dday(x):
    try:
        if data['id'][x.name] == data['id'][x.name - 1]:
            return (data['visit_date'][x.name] - data['visit_date'][x.name-1]).days
        else:
            return 0
    except KeyError:
        return 0

  
  data = pd.read_csv('.\\timeGAN\\merge\\data\\prism.csv')
  
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

  dimensions = ['dday', 'weight', 'height', 'age', 'temp']
  max_ = []
  min_ = []
  for dim in dimensions:
    max_.append(data[dim].max())
    min_.append(data[dim].min())
  def format(data, min_, max_):
    data.interpolate(method = 'linear', inplace=True)
    #np.savez('./data/prism/data_train', data)
    dimensions = ['dday', 'weight', 'height', 'age', 'temp']
    
    
    
    id_unique = data.id.unique()
    features = []
    attributes = []
    gen_flag = []
    for i in id_unique:
        child = np.array(data.loc[data['id'] == i][dimensions])
        if len(child) >= 5:
          if len(child) < 130:
            min_ = [0]*len(dimensions)
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
      new_row = np.multiply(row, np.subtract(max_, min_))#np.apply_along_axis(np.multiply, 0, row, np.subtract(max_, min_))

      new_row = np.add(new_row, min_)#np.apply_along_axis(np.add, 0, row, min_)
      
      new_child.append(np.array(new_row))
    new_child = np.expand_dims(new_child, axis=0)
    try:
      new_data = np.vstack([new_data, new_child])
    except UnboundLocalError:
      new_data = new_child
    
  
  return np.array(new_data)


def add_gen_flag(data_feature, data_gen_flag,
                 sample_len):
    

    if len(data_gen_flag.shape) != 2:
        raise Exception("data_gen_flag should be 2 dimension")

    num_sample, length = data_gen_flag.shape

    data_gen_flag = np.expand_dims(data_gen_flag, 2)

    
    shift_gen_flag = np.concatenate(
        [data_gen_flag[:, 1:, :],
         np.zeros((data_gen_flag.shape[0], 1, 1))],
        axis=1)
    if length % sample_len != 0:
        raise Exception("length must be a multiple of sample_len")
    data_gen_flag_t = np.reshape(
        data_gen_flag,
        [num_sample, int(length / sample_len), sample_len])
    data_gen_flag_t = np.sum(data_gen_flag_t, 2)
    data_gen_flag_t = data_gen_flag_t > 0.5
    data_gen_flag_t = np.repeat(data_gen_flag_t, sample_len, axis=1)
    data_gen_flag_t = np.expand_dims(data_gen_flag_t, 2)
    data_feature = np.concatenate(
        [data_feature,
         shift_gen_flag,
         (1 - shift_gen_flag) * data_gen_flag_t],
        axis=2)

    return data_feature