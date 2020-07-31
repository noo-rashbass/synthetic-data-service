#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
from tensorflow import *
from sklearn.metrics import mean_absolute_error
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import History


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len



#Load the pre-processed data - it randomly divides the whole data set into two parts, 
#one is taken as the original data, the other is taken as the generated data
#They are randomly permuted and normalised somehow (need to have a look at the normalisation)


import numpy as np
with open('/Users/xiaoyanchen/desktop/HDI/TimeGAN/metrics/metrics_v2/stock3ddata.npy', 'rb') as f:
    ori_data = np.load(f)
    generated_data = np.load(f)


def train_val_test_divide(ori_data,train_rate=0.75):
    """Returns the dataset to two sets, can be used for training datasets & validation datasets, each of which    has a input dataset and a groundtruth data for surpervised learning.
  
  Args:
    - ori_data: original data
    - train_rate: the proportion of the ori_data into training data
    
  Returns:
    - x_train: the input training dataset
    - y_train: the ground truth list for the input training dataset
    - x_val: the input validation dataset
    - y_val: the ground truth list for the input validation dataset
  """
    no,seq_len,dim = ori_data.shape
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    val_idx = idx[int(no*train_rate):]
    
    train_ori_data = ori_data[:,:-1,:(dim-1)]
    test_ori_data = ori_data[:,1:,dim-1]
    
    x_train = train_ori_data[train_idx]
    y_train = test_ori_data[train_idx]
    x_val = train_ori_data[val_idx]
    y_val = test_ori_data[val_idx]
    
    return x_train, y_train, x_val, y_val




def last_time_step_mae(Y_true, Y_pred):
   return keras.metrics.MAE(Y_true[:, -1], Y_pred[:, -1])   

# Basic Parameters
no, seq_len, dim = np.asarray(ori_data).shape

# Set maximum sequence length and each sequence length
ori_time, ori_max_seq_len = extract_time(ori_data)
generated_time, generated_max_seq_len = extract_time(ori_data)
max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  

## Builde a post-hoc RNN predictive network 
# Network parameters
hidden_dim = int(dim/2)
iterations = 5000
batch_size = 128
input_size = [None,dim-1]

#Predictive model
predictive_model = Sequential([
    Input(shape = input_size),
    GRU(hidden_dim, return_sequences = True),
    TimeDistributed(Dense(1, activation = "sigmoid"))
])
predictive_model.compile(optimizer = "adam", loss = keras.losses.MeanAbsoluteError(),metrics=[last_time_step_mae])

x_train, y_train, x_val, y_val = train_val_test_divide(ori_data)
history_prediction = predictive_model.fit(x_train,y_train,batch_size = batch_size, epochs = 150, validation_data=(x_val,y_val))



#Store the trained model
from tensorflow.keras.models import model_from_json
# serialize model to JSON
model_json = predictive_model.to_json()
with open("predictive_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
predictive_model.save_weights("predictive_model.h5")




# load json and create model
json_file = open('predictive_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("predictive_model.h5")
 




#Use train_val_test_divide function and set the train_rate to 1.0, we can obtain
#generated_in : the input test dataset from 'generated data'
#generated_test: the ground truth for the input test dataset
#the other two output of train_val_test_divide are trivial in this case
generated_in,generated_test,_1,_2, = train_val_test_divide(generated_data,train_rate=1.0)
#Use the model to predict gien generated_in
generated_prediction = loaded_model.predict(generated_in)




#Compare the predicted result 'generated_prediction' with the ground truth 'generated_test'
MAE_temp = 0
no_gen,seq_gen,dim_gen = np.asarray(generated_data).shape
for i in range(no_gen):
    MAE_temp = MAE_temp + mean_absolute_error(generated_test[i], generated_prediction[i,:,:])
    
predictive_score = MAE_temp / no





#The lower the better
display(predictive_score)

