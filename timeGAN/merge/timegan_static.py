import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# try:
#     physical_devices = tf.config.list_physical_devices('GPU') 
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     pass 
print(tf.__version__)
import numpy as np
import time 
#uncomment line below if not using StackedRNN
from utils import extract_time, rnn_cell, random_generator, random_generator_static, batch_generator_with_static
#uncomment line below to use StackedRNN
#from utilsStacked import extract_time, rnn_cell, random_generator, batch_generator
import warnings
import sys
warnings.filterwarnings("ignore")
tf.keras.backend.set_floatx('float64')

def timegan_static(ori_data, ori_data_static, ori_data_stack, parameters):

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)
    
    
    ori_data = np.array(ori_data)
    ori_data_static = np.array(ori_data_static)
    no_static, dim_static = ori_data_static.shape

    dstack = np.dstack((ori_data, ori_data_stack))
    
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)
    np.save('mix_data_no_seq_2k', dstack)
    
    def MinMaxScaler(data):
        """Min-Max Normalizer.
        
        Args:
        - data: raw data
        
        Returns:
        - norm_data: normalized data
        - min_val: minimum values (for renormalization)
        - max_val: maximum values (for renormalization)
        """    
        min_val = np.nanmin(np.nanmin(data, axis = 0), axis = 0)
        data = data - min_val
        
        max_val = np.nanmax(np.nanmax(data, axis = 0), axis = 0)
        norm_data = data / (max_val + 1e-7)
        
        return norm_data, min_val, max_val
    
    def MinMaxScaler_static(data):
        """Min-Max Normalizer.
        
        Args:
        - data: raw data
        
        Returns:
        - norm_data: normalized data
        - min_val: minimum values (for renormalization)
        - max_val: maximum values (for renormalization)
        """    
        min_val = np.nanmin(data, axis = 0)
        data = data - min_val
        
        max_val = np.nanmax(data, axis = 0)
        norm_data = data / (max_val + 1e-7)
        
        return norm_data, min_val, max_val
  
    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    
    ori_data_static, min_val_static, max_val_static = MinMaxScaler_static(ori_data_static)
    
          
    ## Build a RNN networks          
  
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] 
    num_layers   = parameters['num_layer']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module'] 
    z_dim        = dim
    z_dim_static = dim_static
    gamma        = 1


    def make_embedder ():
        """Embedding network between original feature space to latent space.
        
        Args for model:
        - X: input time-series features
        
        Model returns:
        - H: embeddings
        """
        embedder_model = tf.keras.Sequential(name='embedder')
        embedder_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len,dim)))
        for i in range(num_layers-1):
            embedder_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        embedder_model.add(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))

        return embedder_model



    def make_embedder_static ():  
        """
        Embedder model for static values
        """
       
        embedder_model_static = tf.keras.Sequential(name="embedder_static", layers= [
            
        tf.keras.layers.Dense(hidden_dim, input_shape=(dim_static,)),
        tf.keras.layers.Dense(hidden_dim),
        tf.keras.layers.Dense(hidden_dim),
        tf.keras.layers.Dense(hidden_dim),
        tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)
        
        ])

        return embedder_model_static

    def make_recovery ():   
        """Recovery network from latent space to original space.
        
        Args for model:
        - H: latent representation
        
        Model returns:
        - X_tilde: recovered data
        """      
        recovery_model = tf.keras.Sequential(name='recovery')
        for i in range(num_layers):
            recovery_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        recovery_model.add(tf.keras.layers.Dense(dim, activation='sigmoid'))

        return recovery_model

    def make_recovery_static ():  
        """
        recovery model for static values
        """
            
        recovery_model_static = tf.keras.Sequential(name="recovery_static", layers= [
            
        tf.keras.layers.Dense(dim_static, input_shape=(hidden_dim,)),
        tf.keras.layers.Dense(dim_static),
        tf.keras.layers.Dense(dim_static),
        tf.keras.layers.Dense(dim_static),
        tf.keras.layers.Dense(dim_static, activation=tf.nn.sigmoid)
        
        ])

        return recovery_model_static
  
    def make_generator ():  
        """Generator function: Generate time-series data in latent space.
        
        Args for model:
        - Z: random variables
        
        MOdel returns:
        - E: generated embedding
        """  
        generator_model = tf.keras.Sequential(name='generator')
        generator_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, dim)))
        for i in range(num_layers-1):
            generator_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        generator_model.add(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))


        return generator_model

    def make_generator_static ():  
        """
        generator model for static values
        """
        
        generator_model_static = tf.keras.Sequential(name="generator_static", layers= [
            
            tf.keras.layers.Dense(hidden_dim, input_shape=(dim_static,)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
        ])

        return generator_model_static

    def make_supervisor (): 
        """Generate next sequence using the previous sequence.
        
        Args for model:
        - H: latent representation
        
        Model returns:
        - S: generated sequence based on the latent representations generated by the generator
        """       
        supervisor_model = tf.keras.Sequential(name='supervisor')
        for i in range(num_layers-1):
            supervisor_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim*2)))
        supervisor_model.add(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))

        return supervisor_model
    
    def make_discriminator ():   
        """Recovery network from latent space to original space.
        
        Args for model:
        - H: latent representation
        
        Model returns:
        - Y_hat: classification results between original and synthetic time-series
        """   
        discriminator_model = tf.keras.Sequential(name='discriminator')
        for i in range(num_layers):
            discriminator_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        discriminator_model.add(tf.keras.layers.Dense(1, activation=None))

        return discriminator_model

    def make_discriminator_static ():  
        """
        discirminator model for static values
        """
        
        discriminator_model_static = tf.keras.Sequential(name="discriminator_static", layers= [
            
            tf.keras.layers.Dense(hidden_dim, input_shape=(hidden_dim,)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Dense(1, activation=None),
        ])

        return discriminator_model_static
    
    # make the models
    embedder_model = make_embedder()
    recovery_model = make_recovery()
    generator_model = make_generator()
    supervisor_model = make_supervisor()
    discriminator_model = make_discriminator()

    embedder_model_static = make_embedder_static()
    recovery_model_static = make_recovery_static()
    generator_model_static = make_generator_static()
    discriminator_model_static = make_discriminator_static()


    def get_embedder_T0_loss(X, X_tilde):
        mse = tf.keras.losses.MeanSquaredError() 
        E_loss_T0 = mse(X, X_tilde)
        return E_loss_T0

    def get_embedder_0_loss(X, X_tilde): 
        E_loss_T0 = get_embedder_T0_loss(X, X_tilde)
        E_loss0 = 10*tf.sqrt(E_loss_T0)
        return E_loss0
    
    def get_embedder_loss(X, X_tilde, H, H_hat_supervise):
        """
        computes embedder network loss
        
        Args:
        - X: input time-series features
        - X_tilde: recovered data
        - H: latent representation
        - H_hat_supervise: generated sequence based on the latent representations generated by the generator
        
        Returns:
        - E_loss: embedder loss
        """
        E_loss_T0 = get_embedder_T0_loss(X, X_tilde)
        E_loss0 = 10*tf.sqrt(E_loss_T0) #could use function above
        G_loss_S = get_generator_s_loss(H, H_hat_supervise)
        E_loss = E_loss0 + 0.1*G_loss_S
        return E_loss
    

    def get_generator_s_loss(H, H_hat_supervise):
        """
        computes supervised loss

        Args:
        - H: latent representation
        - H_hat_supervise: generated sequence based on the latent representations generated by the generator
       
        Returns:
        - G_loss_s: supervised loss for generator
        """
        mse = tf.keras.losses.MeanSquaredError()
        G_loss_S = mse(H[:,1:,:], H_hat_supervise[:,:-1,:])
        return G_loss_S

    def get_generator_loss(Y_fake, Y_fake_e, X_hat, X, H, H_hat_supervise):
        """
        computes generator loss for time series variables

        Args:
        - Y_fake: classification results of latent synthetic time-series
        - Y_fake_e: classification results of generated sequence for latent synthetic time-series
        - X_hat: recovered data
        - X: input time-series data
        - H: latent representation
        - H_hat_supervise: generated sequence for latent representation

        Returns:
        - G_loss: generator loss
        - G_loss_U: unsupervised generator loss
        - G_loss_S: supervised generator loss
        - G_loss_V: moments loss for generator
        """
        #1. Adversarial loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)
        G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

        #2. Two Moments
        X = tf.convert_to_tensor(X)
        G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
        G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
        G_loss_V = G_loss_V1 + G_loss_V2

        #3. Supervised loss
        G_loss_S = get_generator_s_loss(H, H_hat_supervise)

        #4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V
        return G_loss, G_loss_U, G_loss_S, G_loss_V
    
    def get_generator_loss_static(Y_fake_e, X_hat, X):
        """
        returns generator loss for static values
        """
        #1. Adversarial loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

        #2. Two Moments
        X = tf.convert_to_tensor(X)
        G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
        G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
        G_loss_V = G_loss_V1 + G_loss_V2

        
        #4. Summation
        G_loss = gamma * G_loss_U_e +  100*G_loss_V
        return G_loss,  G_loss_V
    
    def get_generator_loss_both(Y_fake, Y_fake_e, X_hat, X, H, H_hat_supervise, Y_fake_e_static, X_hat_static, X_static):
        """
        returns generator loss for both static and time series variables
        """
        #1. Adversarial loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)
        G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)
        G_loss_U_e_static = bce(tf.ones_like(Y_fake_e_static), Y_fake_e_static)

        #2. Two Moments
        X = tf.convert_to_tensor(X)
        G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
        G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
        G_loss_V = G_loss_V1 + G_loss_V2

        X_static = tf.convert_to_tensor(X_static)
        G_loss_V1_static = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat_static,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X_static,[0])[1] + 1e-6)))
        G_loss_V2_static = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat_static,[0])[0]) - (tf.nn.moments(X_static,[0])[0])))
        G_loss_V_static = G_loss_V1_static + G_loss_V2_static

        #3. Supervised loss
        G_loss_S = get_generator_s_loss(H, H_hat_supervise)

        #4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V + gamma * G_loss_U_e_static + 100 * G_loss_V_static
        return G_loss, G_loss_U, G_loss_S, G_loss_V, G_loss_V_static

        
    def get_discriminator_loss(Y_real, Y_fake, Y_fake_e):
        """
        computes discrminator loss for time series variables
        
        Args:
        - Y_real: classification results of latent real time-series
        - Y_fake: classification results of latent synthetic time-series
        - Y_fake_e: classification results of generated sequence for latent synthetic time-series
        
        Returns:
        - d_loss: discriminator loss
        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) #loss for cls of latent real data seq
        #default arg for tf.keras.losses.BinaryCrossentropy reduction=losses_utils.ReductionV2.AUTO
        D_loss_real = bce(tf.ones_like(Y_real), Y_real)
        D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake) #loss for cls of latent synthethic data seq
        D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e) #loss for cls of latent synthetic data
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        return D_loss

    def get_discriminator_loss_static(Y_real, Y_fake_e):
        """
        returns discrminator loss for static values
        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) #loss for cls of latent real data seq
        #default arg for tf.keras.losses.BinaryCrossentropy reduction=losses_utils.ReductionV2.AUTO
        D_loss_real = bce(tf.ones_like(Y_real), Y_real)
        D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e) #loss for cls of latent synthetic data
        D_loss = D_loss_real + gamma * D_loss_fake_e
        return D_loss

    def get_discriminator_loss_both(Y_real, Y_fake, Y_fake_e, Y_real_static, Y_fake_e_static):
        """
        returns discrminator loss for both static and temporal
        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) #loss for cls of latent real data seq
        #default arg for tf.keras.losses.BinaryCrossentropy reduction=losses_utils.ReductionV2.AUTO
        D_loss_real = bce(tf.ones_like(Y_real), Y_real)
        D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake) #loss for cls of latent synthethic data seq
        D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e) #loss for cls of latent synthetic data

        D_loss_real_static = bce(tf.ones_like(Y_real_static), Y_real_static)
        D_loss_fake_e_static = bce(tf.zeros_like(Y_fake_e_static), Y_fake_e_static)

        D_loss_temporal = D_loss_real + D_loss_fake + gamma * D_loss_fake_e 
        D_loss_static = D_loss_real_static + D_loss_fake_e_static * gamma
        D_loss = D_loss_temporal + D_loss_static
        return D_loss, D_loss_static

    
    # optimizer
    embedder0_optimizer = tf.keras.optimizers.Adam()
    embedder_optimizer = tf.keras.optimizers.Adam()
    gen_s_optimizer = tf.keras.optimizers.Adam()
    generator_optimizer = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()
    
    embedder0_static_optimizer = tf.keras.optimizers.Adam()
    embedder_static_optimizer = tf.keras.optimizers.Adam()
    generator_static_optimizer = tf.keras.optimizers.Adam()
    discriminator_static_optimizer = tf.keras.optimizers.Adam()


    @tf.function
    def train_step_embedder_static(X_mb):
        """
        trains static embedder model
        """

        with tf.GradientTape() as embedder_static_tape:
            # Embedder & Recovery
            H_mb = embedder_model_static(X_mb)
            X_tilde_mb = recovery_model_static(H_mb)

            embedder_0_loss = get_embedder_0_loss(X_mb, X_tilde_mb)
            emb_vars = embedder_model_static.trainable_variables + recovery_model_static.trainable_variables
            gradients_of_embedder = embedder_static_tape.gradient(embedder_0_loss, emb_vars)
            embedder0_static_optimizer.apply_gradients(zip(gradients_of_embedder, emb_vars))
        
        return embedder_0_loss


    @tf.function
    def train_step_embedder(X_mb):
        """
        trains static embedder model
        """

        with tf.GradientTape() as embedder_tape:
            # Embedder & Recovery
            
            H_mb = embedder_model(X_mb)
            X_tilde_mb = recovery_model(H_mb)
            
            embedder_0_loss = get_embedder_0_loss(X_mb, X_tilde_mb)
            emb_vars = embedder_model.trainable_variables + recovery_model.trainable_variables
            gradients_of_embedder = embedder_tape.gradient(embedder_0_loss, emb_vars)
            embedder0_optimizer.apply_gradients(zip(gradients_of_embedder, emb_vars))
        
        return embedder_0_loss
    

    @tf.function
    def train_step_generator_s(X_mb, X_mb_static):
        """
        supervised training for generator model
        """
        
        with tf.GradientTape() as gen_s_tape:
            
            H_mb = embedder_model(X_mb) 
            H_mb_static = embedder_model_static(X_mb_static)
                        
            H_mb_static = tf.expand_dims(H_mb_static, axis=1)
            H_mb_static = tf.repeat(H_mb_static, seq_len, axis=1)
            
            #Embeddings of both static and temporal features
            H_mb_mix = tf.concat([H_mb, H_mb_static], axis=2)

            # Generator
            H_hat_supervise_mb = supervisor_model(H_mb_mix)
            
            gen_s_loss = get_generator_s_loss(H_mb, H_hat_supervise_mb) 
            gen_s_vars = supervisor_model.trainable_variables
            gradients_of_gen_s = gen_s_tape.gradient(gen_s_loss, gen_s_vars)
            gen_s_optimizer.apply_gradients(zip(gradients_of_gen_s, gen_s_vars))

          

        return gen_s_loss
        

    @tf.function
    def train_step_joint_static(X_mb, X_mb_static, Z_mb):
        """
        joint training for static generator and supervisor model, embedder model
        """
        #train generator ## STATIC Z
        with tf.GradientTape() as gen_tape:
            
            #Embedding
            H_mb_static = embedder_model_static(X_mb_static)
            
            #synthetic embedding
            E_hat_mb = generator_model_static(Z_mb)          
              
            # Synthetic data
            X_hat_mb = recovery_model_static(E_hat_mb)
            
            # Discriminator
            Y_fake_e_mb = discriminator_model_static(E_hat_mb)

            gen_loss, g_loss_v = get_generator_loss_static(Y_fake_e_mb, X_hat_mb, X_mb_static)
            gen_vars = generator_model_static.trainable_variables
            gradients_of_gen = gen_tape.gradient(gen_loss, gen_vars)
            generator_static_optimizer.apply_gradients(zip(gradients_of_gen, gen_vars))

        
        #train embedder
        with tf.GradientTape() as embedder_tape:

            H_mb_static = embedder_model_static(X_mb_static)

            X_tilde_mb = recovery_model_static(H_mb_static) 
                      
            emb_T0_loss = get_embedder_T0_loss(X_mb_static, X_tilde_mb)
            emb_loss = get_embedder_0_loss(X_mb_static, X_tilde_mb) 
            emb_vars = embedder_model_static.trainable_variables + recovery_model_static.trainable_variables
            gradients_of_emb = embedder_tape.gradient(emb_loss, emb_vars)
            embedder_static_optimizer.apply_gradients(zip(gradients_of_emb, emb_vars))
        
        return emb_T0_loss, emb_loss, g_loss_v 

    @tf.function
    def train_step_joint_both(X_mb, X_mb_static, Z_mb, Z_mb_static):
        """
        joint training for both static and temporal generator and supervisor model, embedder model
        """
        #train generator - temporal + static
        with tf.GradientTape() as gen_tape:
            
            #Embeddings
            H_mb = embedder_model(X_mb) 
            H_mb_static_1 = embedder_model_static(X_mb_static)

            H_mb_static = tf.expand_dims(H_mb_static_1, axis=1)
            H_mb_static = tf.repeat(H_mb_static, seq_len, axis=1)

            #Combine static and temporal features
            H_mb_mix = tf.concat([H_mb, H_mb_static], axis=2)
            
            E_hat_mb = generator_model(Z_mb)
            E_hat_mb_static_1 = generator_model_static(Z_mb_static)

            E_hat_mb_static = tf.expand_dims(E_hat_mb_static_1, axis=1)
            E_hat_mb_static = tf.repeat(E_hat_mb_static, seq_len, axis=1)

            #Combine static generator with temporal generator
            E_hat_mb_mix = tf.concat([E_hat_mb, E_hat_mb_static], axis=2)
                        
            H_hat_mb = supervisor_model(E_hat_mb_mix) 
            H_hat_supervise_mb = supervisor_model(H_mb_mix) 

            # Synthetic data
            X_hat_mb = recovery_model(H_hat_mb)
            
            # Discriminator
            Y_fake_mb = discriminator_model(H_hat_mb)
            Y_fake_e_mb = discriminator_model(E_hat_mb)         
              
            ###### STATIC   
            # Synthetic data
            X_hat_mb_static = recovery_model_static(E_hat_mb_static_1)
            
            # Discriminator
            Y_fake_e_mb_static = discriminator_model_static(E_hat_mb_static_1)

            gen_loss, g_loss_u, gen_s_loss, g_loss_v, g_loss_v_static = get_generator_loss_both(Y_fake_mb, Y_fake_e_mb, X_hat_mb, X_mb, H_mb, H_hat_supervise_mb, Y_fake_e_mb_static, X_hat_mb_static, X_mb_static)
            gen_vars = generator_model.trainable_variables + supervisor_model.trainable_variables + generator_model_static.trainable_variables
            gradients_of_gen = gen_tape.gradient(gen_loss, gen_vars)
            generator_optimizer.apply_gradients(zip(gradients_of_gen, gen_vars))
        
        
        #train embedder - temporal
        with tf.GradientTape() as embedder_tape:

            H_mb = embedder_model(X_mb) #recall
            H_mb_static = embedder_model_static(X_mb_static)

            X_tilde_mb = recovery_model(H_mb) 
            
            H_mb_static = tf.expand_dims(H_mb_static, axis=1)
            H_mb_static = tf.repeat(H_mb_static, seq_len, axis=1)

            H_mb_mix = tf.concat([H_mb, H_mb_static], axis=2)
            H_hat_supervise = supervisor_model(H_mb_mix) 

            emb_T0_loss = get_embedder_T0_loss(X_mb, X_tilde_mb)
            emb_loss = get_embedder_loss(X_mb, X_tilde_mb, H_mb, H_hat_supervise) 
            emb_vars = embedder_model.trainable_variables + recovery_model.trainable_variables
            gradients_of_emb = embedder_tape.gradient(emb_loss, emb_vars)
            embedder_optimizer.apply_gradients(zip(gradients_of_emb, emb_vars))
        
        #train embedder - static
        with tf.GradientTape() as embedder_tape:

            H_mb_static = embedder_model_static(X_mb_static)

            X_tilde_mb_static = recovery_model_static(H_mb_static) 
                      
            emb_T0_loss_static = get_embedder_T0_loss(X_mb_static, X_tilde_mb_static)
            emb_loss_static = get_embedder_0_loss(X_mb_static, X_tilde_mb_static) #Not sure which embedder loss to use
            emb_vars_static = embedder_model_static.trainable_variables + recovery_model_static.trainable_variables
            gradients_of_emb_static = embedder_tape.gradient(emb_loss_static, emb_vars_static)
            embedder_static_optimizer.apply_gradients(zip(gradients_of_emb_static, emb_vars_static))
        
        return emb_T0_loss, emb_loss, g_loss_u, gen_s_loss, g_loss_v, emb_T0_loss_static, g_loss_v_static
        
    @tf.function
    def train_step_discriminator_static(X_mb, X_mb_static, Z_mb):
        """
        trains static discriminator model
        """
        
        with tf.GradientTape() as disc_tape:
            
            H_mb_static = embedder_model_static(X_mb_static)

            E_hat_mb = generator_model_static(Z_mb)
            
            # Synthetic data
            X_hat_mb = recovery_model_static(E_hat_mb)
            
            # Discriminator
            Y_real_mb = discriminator_model_static(H_mb_static)
            Y_fake_e_mb = discriminator_model_static(E_hat_mb)

            # Check discriminator loss before updating
            disc_loss = get_discriminator_loss_static(Y_real_mb, Y_fake_e_mb)
            # Train discriminator (only when the discriminator does not work well)
            if (disc_loss > 0.15):
                #disc_loss = get_discriminator_loss(Y_real_mb, Y_fake_mb, Y_fake_e_mb)
                disc_vars = discriminator_model_static.trainable_variables
                gradients_of_disc = disc_tape.gradient(disc_loss, disc_vars)
                discriminator_static_optimizer.apply_gradients(zip(gradients_of_disc, disc_vars))
        
        return disc_loss

    @tf.function
    def train_step_discriminator_both(X_mb, X_mb_static, Z_mb, Z_mb_static):
        """
        trains both static and temporal discriminator model
        """
        
        #training discriminator - temporal + static
        with tf.GradientTape() as disc_tape:
            
            H_mb = embedder_model(X_mb)
            H_mb_static_1 = embedder_model_static(X_mb_static)

            H_mb_static = tf.expand_dims(H_mb_static_1, axis=1)
            H_mb_static = tf.repeat(H_mb_static, seq_len, axis=1)
            H_mb_mix = tf.concat([H_mb, H_mb_static], axis=2)

            E_hat_mb = generator_model(Z_mb)
            E_hat_mb_static_1 = generator_model_static(Z_mb_static)

            E_hat_mb_static = tf.expand_dims(E_hat_mb_static_1, axis=1)
            E_hat_mb_static = tf.repeat(E_hat_mb_static, seq_len, axis=1)
            
            E_hat_mb_mix = tf.concat([E_hat_mb, E_hat_mb_static], axis=2)
            H_hat_mb = supervisor_model(E_hat_mb_mix) 
            
            # Synthetic data
            X_hat_mb = recovery_model(H_hat_mb)
            
            # Discriminator
            Y_fake_mb = discriminator_model(H_hat_mb)
            Y_real_mb = discriminator_model(H_mb)
            Y_fake_e_mb = discriminator_model(E_hat_mb)

            ### Discriminator static
            Y_real_mb_static = discriminator_model_static(H_mb_static_1)
            Y_fake_e_mb_static = discriminator_model_static(E_hat_mb_static_1)

            # Check discriminator loss before updating
            disc_loss, disc_loss_static = get_discriminator_loss_both(Y_real_mb, Y_fake_mb, Y_fake_e_mb, Y_real_mb_static, Y_fake_e_mb_static)
            # Train discriminator (only when the discriminator does not work well)
            if (disc_loss > 0.15):
                disc_vars = discriminator_model.trainable_variables + discriminator_model_static.trainable_variables
                gradients_of_disc = disc_tape.gradient(disc_loss, disc_vars)
                discriminator_optimizer.apply_gradients(zip(gradients_of_disc, disc_vars))
        
        return disc_loss, disc_loss_static

    #timeGAN training
    def train():
        #1. Embedding static network training
        print('Start Static Embedding Network Training')

        for itt in range(iterations):
            # Set mini-batch
            _, X_mb_static, _ = batch_generator_with_static(ori_data, ori_data_static, ori_time, batch_size)
            
            # Train embedder
            step_e_loss = train_step_embedder_static(X_mb_static)
           
            # Checkpoint
            if itt % 1000 == 0:
                print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )

        print('Finish static Embedding Network Training')

        #1. Embedding network training
        
        print('Start Embedding Network Training')
        for itt in range(iterations):
            # Set mini-batch
            X_mb, _, T_mb = batch_generator_with_static(ori_data, ori_data_static, ori_time, batch_size)
            # Train embedder
            step_e_loss = train_step_embedder(X_mb)
           
            # Checkpoint
            if itt % 1000 == 0:
                print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )
        

        
        #2. Training only with supervised loss
        print('Start Training with Supervised Loss Only')

        for itt in range(iterations):
            # Set mini-batch
            X_mb, X_mb_static, T_mb = batch_generator_with_static(ori_data, ori_data_static, ori_time, batch_size)
            # Train generator
            step_gen_s_loss = train_step_generator_s(X_mb, X_mb_static)

            # Checkpoint
            if itt % 1000 == 0:
                print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_gen_s_loss),4)) )

        print('Finish Training with Supervised Loss Only')
        
        # 3. Joint Training
        print('Start Joint Training')

        for itt in range(iterations):

            # Generator training (twice more than discriminator training)
            for kk in range(2):
                # Set mini-batch
                X_mb, X_mb_static, T_mb = batch_generator_with_static(ori_data, ori_data_static, ori_time, batch_size) 
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                Z_mb_static = random_generator_static(batch_size, z_dim_static, T_mb, max_seq_len)
                # Train generator and embedder
                emb_T0_loss, emb_loss, g_loss_u, gen_s_loss, g_loss_v, emb_T0_loss_static, g_loss_v_static = train_step_joint_both(X_mb, X_mb_static, Z_mb, Z_mb_static)
            
            # Discriminator training        
            # Set mini-batch
            X_mb, X_mb_static, T_mb = batch_generator_with_static(ori_data, ori_data_static, ori_time, batch_size)           
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            Z_mb_static = random_generator_static(batch_size, z_dim_static, T_mb, max_seq_len)
            #train discriminator
            d_loss, d_loss_static = train_step_discriminator_both(X_mb, X_mb_static, Z_mb, Z_mb_static)

            # Print multiple checkpoints
            if itt % 200 == 0:
                print('step: '+ str(itt) + '/' + str(iterations) + 
                    ', d_loss: ' + str(np.round(d_loss,4)) + 
                    ', g_loss_u: ' + str(np.round(g_loss_u,4)) + 
                    ', g_loss_s: ' + str(np.round(np.sqrt(gen_s_loss),4)) + 
                    ', g_loss_v: ' + str(np.round(g_loss_v,4)) + 
                    ', e_loss_t0: ' + str(np.round(np.sqrt(emb_T0_loss),4)) +
                    ', d_loss_static: ' + str(np.round(d_loss_static,4)) + 
                    ', g_loss_v_static: ' + str(np.round(g_loss_v_static,4)) )      
        
        print('Finish Joint Training')
        

        ## Synthetic data generation
        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        Z_mb_static = random_generator_static(no, z_dim_static, ori_time, max_seq_len)
        
        # generate in latent dim
        E_hat_generated = generator_model(Z_mb)
        E_hat_generated_static = generator_model_static(Z_mb_static)

        # repeat for seq_len for static values
        E_hat_generated_static_ = tf.expand_dims(E_hat_generated_static, axis=1)
        E_hat_generated_static_ = tf.repeat(E_hat_generated_static_, seq_len, axis=1)
        
        # join static and temporal together
        E_hat_generated_mix = tf.concat([E_hat_generated, E_hat_generated_static_], axis=2)
        
        H_hat_generated = supervisor_model(E_hat_generated_mix)

        # map up to original dimension
        generated_data_curr = recovery_model(H_hat_generated)
        generated_data_curr_static = recovery_model_static(E_hat_generated_static)
        
        generated_data_static = list()

        for i in range(no):
            temp = generated_data_curr_static[i,:]
            generated_data_static.append(temp)
                
        # Renormalization
        generated_data_static = generated_data_static * max_val_static
        generated_data_static = generated_data_static + min_val_static

        generated_data_seq = np.array([[generated_data_static[i] for _ in range(seq_len)] for i in range(no)])

        generated_data = list()

        for i in range(no):
            temp = generated_data_curr[i,:ori_time[i],:]
            generated_data.append(temp)
                
        # Renormalization

        generated_data = generated_data * max_val
        generated_data = generated_data + min_val

        generated_data = np.dstack((generated_data, generated_data_seq))

        return generated_data

    return train()




