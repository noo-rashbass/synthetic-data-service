import tensorflow as tf 
print(tf.__version__)
import numpy as np
#uncomment line below if not using StackedRNN
from utils import extract_time, rnn_cell, random_generator, batch_generator
#uncomment line below to use StackedRNN
#from utilsStacked import extract_time, rnn_cell, random_generator, batch_generator
import warnings
warnings.filterwarnings("ignore")
tf.keras.backend.set_floatx('float64')

def timegan(ori_data, parameters):
    

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


    def make_embedder ():
        """Embedding network between original feature space to latent space.
        
        Args:
        - X: input time-series features
        - T: input time information
        
        Returns:
        - H: embeddings
        """
        embedder_model = tf.keras.Sequential(name='embedder')
        embedder_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len,dim)))
        for i in range(num_layers-1):
            embedder_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        embedder_model.add(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))

        
##        e_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(hidden_dim, activation = tf.nn.tanh, input_shape=(seq_len, hidden_dim)) for _ in range(num_layers-1)])
##        embedder_model = tf.keras.Sequential([
##
##            rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len,dim)),
##                                   
##            tf.keras.layers.RNN(e_cell, return_sequences=True), 
##                       
##            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)
##
##          ])

        return embedder_model

    def make_recovery ():   
        """Recovery network from latent space to original space.
        
        Args:
        - H: latent representation
        - T: input time information
        
        Returns:
        - X_tilde: recovered data
        """     
        recovery_model = tf.keras.Sequential(name='recovery')
        for i in range(num_layers):
            recovery_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        recovery_model.add(tf.keras.layers.Dense(dim, activation='sigmoid'))

##        r_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(hidden_dim, activation = tf.nn.tanh, input_shape=(seq_len, hidden_dim)) for _ in range(num_layers)])
##        recovery_model = tf.keras.Sequential([
##            
##                        
##            tf.keras.layers.RNN(r_cell, return_sequences = True),
##                       
##            tf.keras.layers.Dense(dim, activation=tf.nn.sigmoid)
##
##          ])

        return recovery_model
  
    def make_generator ():  
        """Generator function: Generate time-series data in latent space.
        
        Args:
        - Z: random variables
        - T: input time information
        
        Returns:
        - E: generated embedding
        """ 
        generator_model = tf.keras.Sequential(name='generator')
        generator_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, dim)))
        for i in range(num_layers-1):
            generator_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        generator_model.add(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))

##        e_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(hidden_dim, activation = tf.nn.tanh) for _ in range(num_layers)])
##        generator_model = tf.keras.Sequential([
##                                    
##            tf.keras.layers.RNN(e_cell, return_sequences=True),
##                       
##            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)
##
##          ])


        return generator_model

    def make_supervisor (): 
        """Generate next sequence using the previous sequence.
        
        Args:
        - H: latent representation
        - T: input time information
        
        Returns:
        - S: generated sequence based on the latent representations generated by the generator
        """     
        supervisor_model = tf.keras.Sequential(name='supervisor')
        for i in range(num_layers-1):
            supervisor_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        supervisor_model.add(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))

##        e_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(hidden_dim, activation = tf.nn.tanh, input_shape=(seq_len, hidden_dim)) for _ in range(num_layers-1)])
##        supervisor_model = tf.keras.Sequential([
##                       
##            tf.keras.layers.RNN(e_cell, return_sequences=True),
##                       
##            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)
##
##          ])

        return supervisor_model
    
    def make_discriminator ():   
        """Recovery network from latent space to original space.
        
        Args:
        - H: latent representation
        - T: input time information
        
        Returns:
        - X_tilde: recovered data
        """     
        discriminator_model = tf.keras.Sequential(name='discriminator')
        for i in range(num_layers):
            discriminator_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        discriminator_model.add(tf.keras.layers.Dense(1, activation=None))

##        d_cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(hidden_dim, activation = tf.nn.tanh) for _ in range(num_layers)])
##        discriminator_model = tf.keras.Sequential([
##                       
##            tf.keras.layers.RNN(d_cell),
##                       
##            tf.keras.layers.Dense(1, activation=None)
##
##          ])

        return discriminator_model


    embedder_model = make_embedder()
    recovery_model = make_recovery()
    generator_model = make_generator()
    supervisor_model = make_supervisor()
    discriminator_model = make_discriminator()

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
        returns embedder network loss
        """
        E_loss_T0 = get_embedder_T0_loss(X, X_tilde)
        E_loss0 = 10*tf.sqrt(E_loss_T0) #could use function above
        G_loss_S = get_generator_s_loss(H, H_hat_supervise)
        E_loss = E_loss0 + 0.1*G_loss_S
        return E_loss

    def get_generator_s_loss(H, H_hat_supervise):
        """
        returns supervised loss
        """
        mse = tf.keras.losses.MeanSquaredError()
        G_loss_S = mse(H[:,1:,:], H_hat_supervise[:,:-1,:])
        return G_loss_S

    def get_generator_loss(Y_fake, Y_fake_e, X_hat, X, H, H_hat_supervise):
        """
        returns generator loss
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
    
    def get_discriminator_loss(Y_real, Y_fake, Y_fake_e):
        """
        returns discrminator loss
        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) #loss for cls of latent real data seq
        #default arg for tf.keras.losses.BinaryCrossentropy reduction=losses_utils.ReductionV2.AUTO
        D_loss_real = bce(tf.ones_like(Y_real), Y_real)
        D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake) #loss for cls of latent synthethic data seq
        D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e) #loss for cls of latent synthetic data
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        return D_loss

    # optimizer
    embedder0_optimizer = tf.keras.optimizers.Adam()
    embedder_optimizer = tf.keras.optimizers.Adam()
    gen_s_optimizer = tf.keras.optimizers.Adam()
    generator_optimizer = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step_embedder(X_mb):

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
    def train_step_generator_s(X_mb):
        
        with tf.GradientTape() as gen_s_tape: #, tf.GradientTape() as s_tape:
            
            H_mb = embedder_model(X_mb) #recall
            
            # Generator
            H_hat_supervise_mb = supervisor_model(H_mb)

            gen_s_loss = get_generator_s_loss(H_mb, H_hat_supervise_mb) #hot sure if i should do whole gen loss or only gen_s loss
            gen_s_vars = supervisor_model.trainable_variables #removed possibly useless generator variables.
            #vars = [generator_model.trainable_variables, supervisor_model.trainable_variables]
            gradients_of_gen_s = gen_s_tape.gradient(gen_s_loss, gen_s_vars)
            gen_s_optimizer.apply_gradients(zip(gradients_of_gen_s, gen_s_vars))

            #there's some warning that says gradients do not exist for variables in the generator when minimizing loss

        return gen_s_loss # E_hat_mb, H_hat_mb, H_hat_supervise_mb,  #,generator_model, supervisor_model

    @tf.function
    def train_step_joint(X_mb, Z_mb):
        #train generator
        with tf.GradientTape() as gen_tape:
            # Generator
            #not sure if i should call these generators and supervisors again
            #because returning models from train_step_generator_s and getting trainable variables does not work?
            #so called it again here
            H_mb = embedder_model(X_mb) #recall
            E_hat_mb = generator_model(Z_mb) #is this a recall?
            H_hat_mb = supervisor_model(E_hat_mb) #recall
            H_hat_supervise_mb = supervisor_model(H_mb) #recall

            # Synthetic data
            X_hat_mb = recovery_model(H_hat_mb)
            
            # Discriminator
            Y_fake_mb = discriminator_model(H_hat_mb)
            Y_real_mb = discriminator_model(H_mb)
            Y_fake_e_mb = discriminator_model(E_hat_mb)

            gen_loss, g_loss_u, gen_s_loss, g_loss_v = get_generator_loss(Y_fake_mb, Y_fake_e_mb, X_hat_mb, X_mb, H_mb, H_hat_supervise_mb)
            gen_vars = generator_model.trainable_variables + supervisor_model.trainable_variables
            gradients_of_gen = gen_tape.gradient(gen_loss, gen_vars)
            generator_optimizer.apply_gradients(zip(gradients_of_gen, gen_vars))

        #train embedder
        with tf.GradientTape() as embedder_tape:

            H_mb = embedder_model(X_mb) #recall

            X_tilde_mb = recovery_model(H_mb) 

            H_hat_supervise = supervisor_model(H_mb) #called in order to get emb_loss
            
            #not sure if this should be E_loss or E_loss_T0 
            #i think we are minimizing E_loss but printing out E_loss_T0??
            emb_T0_loss = get_embedder_T0_loss(X_mb, X_tilde_mb)
            emb_loss = get_embedder_loss(X_mb, X_tilde_mb, H_mb, H_hat_supervise) 
            emb_vars = embedder_model.trainable_variables + recovery_model.trainable_variables
            gradients_of_emb = embedder_tape.gradient(emb_loss, emb_vars)
            embedder_optimizer.apply_gradients(zip(gradients_of_emb, emb_vars))
        
        return emb_T0_loss, emb_loss, g_loss_u, gen_s_loss, g_loss_v #H_hat_mb, E_hat_mb, 

    @tf.function
    def train_step_discriminator(X_mb, Z_mb):
        
        with tf.GradientTape() as disc_tape:
            
            H_mb = embedder_model(X_mb) #recall
            E_hat_mb = generator_model(Z_mb) #recall
            H_hat_mb = supervisor_model(E_hat_mb) #recall
            
            # Synthetic data
            X_hat_mb = recovery_model(H_hat_mb)
            
            # Discriminator
            Y_fake_mb = discriminator_model(H_hat_mb)
            Y_real_mb = discriminator_model(H_mb)
            Y_fake_e_mb = discriminator_model(E_hat_mb)

            # Check discriminator loss before updating
            disc_loss = get_discriminator_loss(Y_real_mb, Y_fake_mb, Y_fake_e_mb)
            # Train discriminator (only when the discriminator does not work well)
            if (disc_loss > 0.15):
                #disc_loss = get_discriminator_loss(Y_real_mb, Y_fake_mb, Y_fake_e_mb)
                disc_vars = discriminator_model.trainable_variables
                gradients_of_disc = disc_tape.gradient(disc_loss, disc_vars)
                discriminator_optimizer.apply_gradients(zip(gradients_of_disc, disc_vars))
        
        return disc_loss

    #timeGAN training
    def train():
        #1. Embedding network training
        print('Start Embedding Network Training')

        for itt in range(1):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size) 
            # Train embedder
            step_e_loss = train_step_embedder(X_mb)
           
            # Checkpoint
            if itt % 1 == 0:
                print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )

        print('Finish Embedding Network Training')
        
        #@. Training only with supervised loss
        print('Start Training with Supervised Loss Only')

        for itt in range(iterations):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            
            # Train generator
            step_gen_s_loss = train_step_generator_s(X_mb)

            # Checkpoint
            if itt % 1 == 0:
                print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_gen_s_loss),4)) )

        print('Finish Training with Supervised Loss Only')

        # 3. Joint Training
        print('Start Joint Training')

        for itt in range(iterations):
            # Generator training (twice more than discriminator training)
            for kk in range(2):
                # Set mini-batch
                X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size) 
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                # Train generator and embedder
                emb_T0_loss, emb_loss, g_loss_u, gen_s_loss, g_loss_v = train_step_joint(X_mb, Z_mb)

            # Discriminator training        
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            #train discriminator
            d_loss = train_step_discriminator(X_mb, Z_mb)

            # Print multiple checkpoints
            if itt % 1 == 0:
                print('step: '+ str(itt) + '/' + str(iterations) + 
                    ', d_loss: ' + str(np.round(d_loss,4)) + 
                    ', g_loss_u: ' + str(np.round(g_loss_u,4)) + 
                    ', g_loss_s: ' + str(np.round(np.sqrt(gen_s_loss),4)) + 
                    ', g_loss_v: ' + str(np.round(g_loss_v,4)) + 
                    ', e_loss_t0: ' + str(np.round(np.sqrt(emb_T0_loss),4))  )
        
        print('Finish Joint Training')

        ## Synthetic data generation
        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        E_hat_generated = generator_model(Z_mb)
        H_hat_generated = supervisor_model(E_hat_generated)
        generated_data_curr = recovery_model(H_hat_generated)
        
        generated_data = list()

        for i in range(no):
            temp = generated_data_curr[i,:ori_time[i],:]
            generated_data.append(temp)
                
        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val
    
        return generated_data
        
    return train()


####TESTING####

# from data_loading import real_data_loading, sine_data_generation

# data_name = 'stock'
# seq_len = 9

# if data_name in ['stock', 'energy']:
#  ori_data = real_data_loading(data_name, seq_len)
# elif data_name == 'sine':
#   # Set number of samples and its dimensions
#   no, dim = 15, 5
#   ori_data = sine_data_generation(no, seq_len, dim)
    
# print(data_name + ' dataset is ready.')

# ## Newtork parameters
# parameters = dict()

# parameters['module'] = 'lstm' 
# parameters['hidden_dim'] = 4
# parameters['num_layer'] = 3
# parameters['iterations'] = 10
# parameters['batch_size'] = 7

# generated_data = timegan(ori_data, parameters)
# print('Finish Synthetic Data Generation')
# #print(generated_data)

