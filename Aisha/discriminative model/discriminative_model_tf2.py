#!/usr/bin/env python
# coding: utf-8

# In[58]:


import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


# In[64]:


#Load the pre-processed data from the data preprocessing.py, which should be run firstly
#The npy file should contian two np.array objects, each of size (number of data, sequence length, column dimension)
with open('/stock3ddata.npy', 'rb') as f:
    ori_data = np.load(f)
    generated_data = np.load(f)


# In[65]:


# Basic Parameters
no, seq_len, dim = np.asarray(ori_data).shape 
no_g,seq_len_g,_ = np.asarray(generated_data).shape

# Set maximum sequence length and each sequence length
ori_time, ori_max_seq_len = extract_time(ori_data)
generated_time, generated_max_seq_len = extract_time(ori_data)
max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  

## Builde a post-hoc RNN discriminator network 
# Network parameters
hidden_dim = int(dim/2)
iterations = 2000
batch_size = 128
input_size = [None,dim]


# In[72]:


def mix_divide(ori_data, gen_data, rate=(0.65,0.2,0.15)):
    """Returns Maximum sequence length and each sequence length.
  
    Args:
    - ori_data: the real data
    - gen_data: the generated data
    - rate: a tuple of size 3, consiting of the proportions of data used to train, validate and test

    Returns:
    - train_data: the data used to train the model
    - train_label: 0 or 1 mask for the train_data indicating whether it is real or generated data
    - val_data: the data used to validate
    - val_label: mask for the val_data
    - test_data: the data used to test the model
    - test_label: mask for the test_data
    """
    no = len(ori_data)
    no_g = len(gen_data)
    n = no + no_g
    ones = np.ones(no)
    zeros = np.zeros(no_g)
    idx = np.random.permutation(n)
    
    ori_gen = np.concatenate([ori_data,gen_data],axis=0)
    labels = np.concatenate([ones,zeros])
    ori_gen = ori_gen[idx]
    labels = labels[idx]
    
    train_data = ori_gen[:int(n*rate[0])]
    train_label = labels[:int(n*rate[0])]
    val_data = ori_gen[int(n*rate[0]):int(n*(rate[0]+rate[1]))]
    val_label = labels[int(n*rate[0]):int(n*(rate[0]+rate[1]))]
    test_data = ori_gen[int(n*(1-rate[2])):]
    test_label = labels[int(n*(1-rate[2])):]
    return train_data,train_label,val_data,val_label,test_data,test_label

td,tl,vd,vl,ted,tel = mix_divide(ori_data,generated_data)


# In[93]:


from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import activations
from tensorflow.keras.callbacks import History
def discriminative_model(input_size=input_size):
    inputs1 = Input(shape = input_size)
    GRU_output_sequence, GRU_last_state = GRU(hidden_dim, return_sequences = True, return_state = True)(inputs1)
    #Dense1 is the y_hat_logit in the original code
    Dense1 = Dense(1)(GRU_last_state)
    #Acti1 is the y_hat in the original code
    #It is very odd that the original code seems to compare the result of Dense1 with the one-zero label 
    #while using Acti1 as the prediction result, but it doesn't make sense to me
    #I do what I think to be the right thing here - use Acti1 result as the prediction result
    Acti1 = Activation(activations.sigmoid)(Dense1)      
    
    model = Model(inputs = inputs1, outputs = [Acti1])
    
    def d_loss(y_logit,y_label):
        y_logit = tf.convert_to_tensor(y_logit,dtype=tf.float32)
        y_label = tf.convert_to_tensor(y_label,dtype = tf.float32)
        dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit, labels = y_label))
        return dloss
                              
    model.compile(optimizer = "adam", loss = d_loss)
    return model                          
                                                                                                        
                                                                                                        


# In[96]:


dmodel = discriminative_model()
history_dmodel = dmodel.fit(td,tl,batch_size = batch_size,epochs = 100,validation_data = (vd,vl))


# In[85]:


#Store the trained model
from tensorflow.keras.models import model_from_json
# serialize model to JSON
model_json = dmodel.to_json()
with open("discriminative_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
dmodel.save_weights("discriminative_model.h5")


# In[86]:


# load json and create model
json_file = open('discriminative_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("discriminative_model.h5")


# In[90]:


# "real" or "fake" predicted by the model on the test_data
test_prediction = loaded_model.predict(ted)


# In[91]:


acc = accuracy_score(tel, (test_prediction>0.5))
discriminative_score = np.abs(0.5-acc)


# In[92]:


#criteria
display(discriminative_score)

