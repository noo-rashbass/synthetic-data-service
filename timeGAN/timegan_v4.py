import tensorflow as tf 
print(tf.__version__)
import numpy as np
from utils4 import extract_time, rnn_cell, random_generator, batch_generator

def timegan(ori_data, parameters):
    
    #reset default graph?

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    
    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)
  
    def MinMaxScaler(data):
        """Min-Max Normalizer.
        
        Args:
        - data: raw data
        
        Returns:
        - norm_data: normalized data
        - min_val: minimum values (for renormalization)
        - max_val: maximum values (for renormalization)
        """    
        min_val = np.min(np.min(data, axis = 0), axis = 0)
        data = data - min_val
        
        max_val = np.max(np.max(data, axis = 0), axis = 0)
        norm_data = data / (max_val + 1e-7)
        
        return norm_data, min_val, max_val
  
    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
              
    ## Build a RNN networks          
  
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] 
    num_layers   = parameters['num_layer']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module'] 
    z_dim        = dim
    gamma        = 1

    # Input place holders
    # tf.keras.backend.placeholder
    #tf.Variable
    #https://stackoverflow.com/questions/58986126/replacing-placeholder-for-tensorflow-v2
    #X = tf.keras.Input(shape=(max_seq_len, dim), dtype="float32", name="myinput_x")
    #Z = tf.keras.Input(shape=(max_seq_len, dim), dtype="float32", name="myinput_z")
    #T = tf.keras.Input(shape=(), dtype="int32", name="myinput_t")

    def embedder ():
        """Embedding network between original feature space to latent space.
        
        Args:
        - X: input time-series features
        - T: input time information
        
        Returns:
        - H: embeddings
        """
        #e_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(hidden_dim, activation = tf.nn.tanh) for _ in range(num_layers)])
        e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])

        model = tf.keras.Sequential([
                #tf.keras.layers.Embedding(24, 24, batch_input_shape=[batch_size, None]),            
                tf.keras.layers.RNN(e_cell, dtype=tf.float32),            
                tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)
            ])

        return model

    def recovery (H):   
        """Recovery network from latent space to original space.
        
        Args:
        - H: latent representation
        - T: input time information
        
        Returns:
        - X_tilde: recovered data
        """     
        r_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        
        model = tf.keras.Sequential([
                tf.keras.layers.RNN(r_cell, H, dtype=tf.float32),        
                tf.keras.layers.Dense(dim, activation=tf.nn.sigmoid)
            ])

        return model

    # Embedder & Recovery
    embedder_model = embedder()
    H = embedder_model(X)
    #X_tilde = recovery(H)

    print(embedder_model.summary())

####TESTING####

from data_loading import real_data_loading, sine_data_generation

data_name = 'sine'
seq_len = 5

if data_name in ['stock', 'energy']:
  ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
  # Set number of samples and its dimensions
  no, dim = 50, 2
  ori_data = sine_data_generation(no, seq_len, dim)
    
print(data_name + ' dataset is ready.')

## Newtork parameters
parameters = dict()

parameters['module'] = 'lstm' 
parameters['hidden_dim'] = 6
parameters['num_layer'] = 3
parameters['iterations'] = 10
parameters['batch_size'] = 4

timegan(ori_data, parameters)