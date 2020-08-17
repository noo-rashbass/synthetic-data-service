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


def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.nanmin(data, 0)
  denominator = np.nanmax(data, 0) - np.nanmin(data, 0)
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
  assert data_name in ['stock','energy', 'prism', 'sine_sampling']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'prism':
    #so that the interpolate method can be used to fillnas
    ori_data = pd.read_csv('data/dt_10visits_noid.csv')
    ori_data.interpolate(method = 'linear', inplace=True)
    ori_data = np.asarray(ori_data)
  elif data_name == "sine_sampling":
    ori_data = np.loadtxt('data/sine_30000_sampling50.csv', delimiter= ",", skiprows=1)
  

  # Flip the data to make chronological data
  if data_name == "stock" or data_name == "energy":
    ori_data = ori_data[::-1]
  # Normalize the data
  #ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data), seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return temp_data

#currently not used as tgan can't really generate good data with varying time length
def real_data_loading_prism_dt():
  """
  loads data with different visit length
  """

  data = pd.read_csv('data/time_patients_25to50visits.csv')

  min_val = data.min()
  max_val = data.max()

  scaled_data = (data - min_val) / (max_val - min_val + 1e-7)
  scaled_data = scaled_data.fillna(-1)

  id_unique = scaled_data.id.unique()

  row_list =[] 
  j = 0
  #iterate over id
  for i in id_unique:  
      id_list = []
      count = 0
      for index, rows in scaled_data[j:j+51].iterrows(): 
          # get row of the same id 
          if rows.id == i:
              count +=1
              my_list =[rows.day_num, rows.height, rows.weight, rows.temp, rows.vomit_dur, rows.cough_dur]
        
              #append a row to its id list
              id_list.append(my_list) 
      #append the id list to the whole patients list
      row_list.append((id_list))
      j +=count
  
  min_val = min_val.drop('id')
  max_val = max_val.drop('id') 

  return np.asarray(row_list), min_val, max_val