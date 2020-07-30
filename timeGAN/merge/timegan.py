import tensorflow as tf 
print(tf.__version__)
import numpy as np
#uncomment line below if not using StackedRNN
from utils import extract_time, rnn_cell, random_generator, batch_generator
#uncomment line below to use StackedRNN
#from utils5 import extract_time, rnn_cell, random_generator, batch_generator
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
        min_val = np.nanmin(np.nanmin(data, axis = 0), axis = 0)
        data = data - min_val
        
        max_val = np.nanmax(np.nanmax(data, axis = 0), axis = 0)
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
    mask_value   = 0

    def make_embedder ():
        """Embedding network between original feature space to latent space.
        
        Args:
        - X: input time-series features
        - T: input time information
        
        Returns:
        - H: embeddings
        """
        embedder_model = tf.keras.Sequential(name='embedder')
        embedder_model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(seq_len,dim)))
        embedder_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len,dim)))
        for i in range(num_layers-1):
            embedder_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        embedder_model.add(tf.keras.layers.Dense(hidden_dim, activation='sigmoid'))

        
        # e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim, input_shape=(seq_len, hidden_dim)) for _ in range(num_layers-1)])
    
        # embedder_model = tf.keras.Sequential([
            #needs some code for the first layer with diff input size   
            # but this does not solve the "multiple" output shape issue??               
        #    tf.keras.layers.RNN(e_cell, return_sequences=True), 
                       
        #    tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)

        # ])

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
        recovery_model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(seq_len, hidden_dim)))
        for i in range(num_layers):
            recovery_model.add(rnn_cell(module_name, hidden_dim, return_sequences=True, input_shape=(seq_len, hidden_dim)))
        recovery_model.add(tf.keras.layers.Dense(dim, activation='sigmoid'))

        # r_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        # recovery_model = tf.keras.Sequential([
                                   
        #     tf.keras.layers.RNN(r_cell, return_sequences=True), 
                       
        #     tf.keras.layers.Dense(dim, activation=tf.nn.sigmoid)

        #   ])

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

        # g_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        # generator_model = tf.keras.Sequential([
                                   
        #     tf.keras.layers.RNN(g_cell, return_sequences=True), 
                       
        #     tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)

        #   ])

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

        # s_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
        # supervisor_model = tf.keras.Sequential([
                                   
        #     tf.keras.layers.RNN(s_cell, return_sequences=True), 
                       
        #     tf.keras.layers.Dense(hidden_dim, activation=tf.nn.sigmoid)

        #   ])

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

        # d_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        # discriminator_model = tf.keras.Sequential([
                                   
        #     tf.keras.layers.RNN(d_cell, return_sequences=True), 
                       
        #     tf.keras.layers.Dense(dim, activation=None)

        #   ])

        return discriminator_model


    embedder_model = make_embedder()
    recovery_model = make_recovery()
    generator_model = make_generator()
    supervisor_model = make_supervisor()
    discriminator_model = make_discriminator()
    print(discriminator_model.summary())

    def get_embedder_T0_loss(X, X_tilde, mask_slice):
        """
        returns embedder_T0 loss
        Args: 
        - X: input time-series information that is masked
        - X_tilde: X that has undergone embedding + recovery and is masked
        - mask_slice: (batch_size * seq_len * 1) tensor which contains information about masked rows
        returns:
        - E_loss_T0 : Scalar embedder_T0 loss
        """
        #mse = tf.keras.losses.MeanSquaredError() #this automatically does reduction from array to scalar
        mse_loss = tf.keras.losses.mean_squared_error(X, X_tilde) #this is still an array and not reduced
        #reduce array to scalar
        #take mean over number of non-masked rows (not seq_length)
        E_loss_T0 = tf.reduce_sum(mse_loss)/ tf.reduce_sum(mask_slice)
        return E_loss_T0

    def get_embedder_0_loss(X, X_tilde, mask_slice): 
        E_loss_T0 = get_embedder_T0_loss(X, X_tilde, mask_slice)
        E_loss0 = 10*tf.sqrt(E_loss_T0)
        return E_loss0
    
    def get_embedder_loss(X, X_tilde, H, H_hat_supervise, mask_slice):
        """
        returns embedder network loss
        """
        E_loss_T0 = get_embedder_T0_loss(X, X_tilde, mask_slice)
        E_loss0 = 10*tf.sqrt(E_loss_T0) #could use function above
        G_loss_S = get_generator_s_loss(H, H_hat_supervise, mask_slice)
        E_loss = E_loss0 + 0.1*G_loss_S
        return E_loss

    def get_generator_s_loss(H, H_hat_supervise, mask_slice):
        """
        returns supervised loss
        """
        #mse = tf.keras.losses.MeanSquaredError()
        #G_loss_S = mse(H[:,1:,:], H_hat_supervise[:,:-1,:])
        mse_loss = tf.keras.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
        G_loss_S = tf.reduce_sum(mse_loss)/ tf.reduce_sum(mask_slice)
        return G_loss_S

    def get_generator_loss(Y_fake, Y_fake_e, X_hat, X, H, H_hat_supervise, mask_slice):
        """
        returns generator loss
        """
        #1. Adversarial loss
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)
        # G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

        # i think this does what we want?
        bce_loss_y_fake = tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake), Y_fake, from_logits=True)  * tf.squeeze(mask_slice)
        G_loss_U = tf.reduce_sum(bce_loss_y_fake)/ tf.reduce_sum(mask_slice)
        bce_loss_y_fake_e = tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True)  * tf.squeeze(mask_slice)
        G_loss_U_e = tf.reduce_sum(bce_loss_y_fake_e)/ tf.reduce_sum(mask_slice)

        #2. Two Moments
        X = tf.convert_to_tensor(X)
        #G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
        #G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))

        #calculate mean- equivalent to masked version of tf.nn.moments(X_hat,[0])[0]
        #calculate variance- equivalent to masked version of tf.nn.moments(X_hat,[0])[1]
        mean_X_hat = tf.reduce_sum(X_hat,[0])/ (tf.reduce_sum(mask_slice,[0]) + 1e-6)
        squared_X_hat = tf.square(X_hat - mean_X_hat) * mask_slice
        variance_X_hat = (tf.reduce_sum(squared_X_hat, [0])/ (tf.reduce_sum(mask_slice, [0])+1e-6)) #sample variance (biased)

        mean_X = tf.reduce_sum(X,[0])/ (tf.reduce_sum(mask_slice,[0]) + 1e-6)
        squared_X = tf.square(X - mean_X) * mask_slice
        variance_X = (tf.reduce_sum(squared_X, [0])/ (tf.reduce_sum(mask_slice, [0])+1e-6)) #sample variance (biased)

        #get num unmasked value for reduced mean = total_num_of_values (seq_len*dim) - num_values_in_completely_masked_rows (rows that are masked in all patients in the batch * dim)
        num_unmasked_values = tf.reduce_sum(tf.clip_by_value(tf.reduce_sum(mask_slice, [2,0]), clip_value_min=0, clip_value_max=1)) * dim 
        G_loss_V1 = tf.reduce_sum(tf.abs(tf.sqrt(variance_X_hat) - tf.sqrt(variance_X)))/(num_unmasked_values + 1e-6)
        G_loss_V2 = tf.reduce_sum(tf.abs(mean_X_hat - mean_X))/ (num_unmasked_values + 1e-6)

        G_loss_V = G_loss_V1 + G_loss_V2

        #3. Supervised loss
        G_loss_S = get_generator_s_loss(H, H_hat_supervise, mask_slice)

        #4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V
        return G_loss, G_loss_U, G_loss_S, G_loss_V
    
    def get_discriminator_loss(Y_real, Y_fake, Y_fake_e, mask_slice):
        """
        returns discrminator loss
        """
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) #loss for cls of latent real data seq
        #default arg for tf.keras.losses.BinaryCrossentropy reduction=losses_utils.ReductionV2.AUTO
        # D_loss_real = bce(tf.ones_like(Y_real), Y_real)
        # D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake) #loss for cls of latent synthethic data seq
        # D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e) #loss for cls of latent synthetic data

        #following method in adversarial loss, i think this is what we want?
        bce_loss_y_real = tf.keras.losses.binary_crossentropy(tf.ones_like(Y_real), Y_real, from_logits=True)  * tf.squeeze(mask_slice)
        D_loss_real = tf.reduce_sum(bce_loss_y_real)/ tf.reduce_sum(mask_slice)
        bce_loss_y_fake = tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake), Y_fake, from_logits=True)  * tf.squeeze(mask_slice)
        D_loss_fake = tf.reduce_sum(bce_loss_y_fake)/ tf.reduce_sum(mask_slice)
        bce_loss_y_fake_e = tf.keras.losses.binary_crossentropy(tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True)  * tf.squeeze(mask_slice)
        D_loss_fake_e = tf.reduce_sum(bce_loss_y_fake_e)/ tf.reduce_sum(mask_slice)
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

            #get a mask by looking at the first column of X_mb
            mask_slice = tf.slice(X_mb, [0,0,0], [batch_size, seq_len, 1]) #first column as mask slice
            mask_val = tf.ones([batch_size, seq_len,1], dtype=tf.float64) * mask_value
            mask_slice = (mask_slice != mask_val) #False means masked
            mask_slice = tf.cast(mask_slice, tf.float64)
            X_mb = tf.multiply(X_mb, mask_slice) #masking

            # Embedder & Recovery
            H_mb = embedder_model(X_mb)
            H_mb = tf.multiply(H_mb, mask_slice)  #masking

            X_tilde_mb = recovery_model(H_mb)
            X_tilde_mb = tf.multiply(X_tilde_mb, mask_slice) #masking

            #should i minimize embedder_0_loss but print out e_loss_T0?
            embedder_0_loss = get_embedder_0_loss(X_mb, X_tilde_mb, mask_slice) #minimize
            embedder_T0_loss = get_embedder_T0_loss(X_mb, X_tilde_mb, mask_slice) #print
            emb_vars = embedder_model.trainable_variables + recovery_model.trainable_variables
            gradients_of_embedder = embedder_tape.gradient(embedder_0_loss, emb_vars)
            embedder0_optimizer.apply_gradients(zip(gradients_of_embedder, emb_vars))

            #tf.print(X_mb)
            #tf.print(X_tilde_mb)
        
        return embedder_T0_loss

    @tf.function
    def train_step_generator_s(X_mb, Z_mb):
        
        with tf.GradientTape() as gen_s_tape: #, tf.GradientTape() as s_tape:

            #get a mask slice for masked value in X_mb
            mask_slice = tf.slice(X_mb, [0,0,0], [batch_size, seq_len, 1]) #first column as mask slice
            mask_val = tf.ones([batch_size, seq_len,1], dtype=tf.float64) * mask_value
            mask_slice = (mask_slice != mask_val) #False means masked
            mask_slice = tf.cast(mask_slice, tf.float64)
            X_mb = tf.multiply(X_mb, mask_slice) #masking
            
            H_mb = embedder_model(X_mb) #recall
            H_mb = tf.multiply(H_mb, mask_slice) #masking
            
            #masking to the input Z_mb (not sure if i should mask this?)
            #if i don't mask it, there might be an issue when comparing masked rows of X and unmasked rows of Z when calculating loss
            Z_mb = tf.multiply(Z_mb, mask_slice)

            # Generator
            E_hat_mb = generator_model(Z_mb)
            E_hat_mb = tf.multiply(E_hat_mb, mask_slice) #masking
            H_hat_mb = supervisor_model(E_hat_mb)
            H_hat_mb = tf.multiply(H_hat_mb, mask_slice) #masking
            H_hat_supervise_mb = supervisor_model(H_mb)
            H_hat_supervise_mb = tf.multiply(H_hat_supervise_mb, mask_slice)

            gen_s_loss = get_generator_s_loss(H_mb, H_hat_supervise_mb, mask_slice) #hot sure if i should do whole gen loss or only gen_s loss
            gen_s_vars = generator_model.trainable_variables + supervisor_model.trainable_variables 
            #vars = [generator_model.trainable_variables, supervisor_model.trainable_variables]
            gradients_of_gen_s = gen_s_tape.gradient(gen_s_loss, gen_s_vars)
            gen_s_optimizer.apply_gradients(zip(gradients_of_gen_s, gen_s_vars))

            #there's some warning that says gradients do not exist for variables in the generator when minimizing loss

        return gen_s_loss # E_hat_mb, H_hat_mb, H_hat_supervise_mb,  #,generator_model, supervisor_model

    @tf.function
    def train_step_joint(X_mb, Z_mb):
        #train generator
        with tf.GradientTape() as gen_tape:

            #get a mask slice for masked value in X_mb
            mask_slice = tf.slice(X_mb, [0,0,0], [batch_size, seq_len, 1]) #first column as mask slice
            mask_val = tf.ones([batch_size, seq_len,1], dtype=tf.float64) * mask_value
            mask_slice = (mask_slice != mask_val) #False means masked
            mask_slice = tf.cast(mask_slice, tf.float64)
            X_mb = tf.multiply(X_mb, mask_slice) #masking
            Z_mb = tf.multiply(Z_mb, mask_slice) #masking

            # Generator
            #not sure if i should call these generators and supervisors again
            #because returning models from train_step_generator_s and getting trainable variables does not work?
            #so called it again here
            H_mb = embedder_model(X_mb) #recall
            H_mb = tf.multiply(H_mb, mask_slice) #masking
            E_hat_mb = generator_model(Z_mb) #is this a recall?
            E_hat_mb = tf.multiply(E_hat_mb, mask_slice) #masking
            H_hat_mb = supervisor_model(E_hat_mb) #recall
            H_hat_mb = tf.multiply(H_hat_mb, mask_slice) #masking
            H_hat_supervise_mb = supervisor_model(H_mb) #recall
            H_hat_supervise_mb = tf.multiply(H_hat_supervise_mb, mask_slice)

            # Synthetic data
            X_hat_mb = recovery_model(H_hat_mb)
            X_hat_mb = tf.multiply(X_hat_mb, mask_slice)
            
            # Discriminator
            Y_fake_mb = discriminator_model(H_hat_mb)
            Y_fake_mb = tf.multiply(Y_fake_mb, mask_slice)
            Y_real_mb = discriminator_model(H_mb)
            Y_real_mb = tf.multiply(Y_real_mb, mask_slice)
            Y_fake_e_mb = discriminator_model(E_hat_mb)
            Y_fake_e_mb = tf.multiply(Y_fake_e_mb, mask_slice)

            gen_loss, g_loss_u, gen_s_loss, g_loss_v = get_generator_loss(Y_fake_mb, Y_fake_e_mb, X_hat_mb, X_mb, H_mb, H_hat_supervise_mb, mask_slice)
            gen_vars = generator_model.trainable_variables + supervisor_model.trainable_variables
            gradients_of_gen = gen_tape.gradient(gen_loss, gen_vars)
            generator_optimizer.apply_gradients(zip(gradients_of_gen, gen_vars))
        
        #train embedder
        with tf.GradientTape() as embedder_tape:

            #get a mask slice for masked value in X_mb
            mask_slice = tf.slice(X_mb, [0,0,0], [batch_size, seq_len, 1]) #first column as mask slice
            mask_val = tf.ones([batch_size, seq_len,1], dtype=tf.float64) * mask_value
            mask_slice = (mask_slice != mask_val) #False means masked
            mask_slice = tf.cast(mask_slice, tf.float64)
            X_mb = tf.multiply(X_mb, mask_slice) #masking

            H_mb = embedder_model(X_mb) #recall
            H_mb = tf.multiply(H_mb, mask_slice) #masking

            X_tilde_mb = recovery_model(H_mb)
            X_tilde_mb = tf.multiply(X_tilde_mb, mask_slice) #masking 
            H_hat_supervise = supervisor_model(H_mb) #called in order to get emb_loss
            H_hat_supervise = tf.multiply(H_hat_supervise, mask_slice)
            
            #not sure if this should be E_loss or E_loss_T0 
            #i think we are minimizing E_loss but printing out E_loss_T0??
            emb_T0_loss = get_embedder_T0_loss(X_mb, X_tilde_mb, mask_slice)
            emb_loss = get_embedder_loss(X_mb, X_tilde_mb, H_mb, H_hat_supervise, mask_slice) 
            emb_vars = embedder_model.trainable_variables + recovery_model.trainable_variables
            gradients_of_emb = embedder_tape.gradient(emb_loss, emb_vars)
            embedder_optimizer.apply_gradients(zip(gradients_of_emb, emb_vars))
        
        return emb_T0_loss, emb_loss, g_loss_u, gen_s_loss, g_loss_v #H_hat_mb, E_hat_mb, 

    @tf.function
    def train_step_discriminator(X_mb, Z_mb):
        
        with tf.GradientTape() as disc_tape:

            #get a mask slice for masked value in X_mb
            mask_slice = tf.slice(X_mb, [0,0,0], [batch_size, seq_len, 1]) #first column as mask slice
            mask_val = tf.ones([batch_size, seq_len,1], dtype=tf.float64) * mask_value
            mask_slice = (mask_slice != mask_val) #False means masked
            mask_slice = tf.cast(mask_slice, tf.float64)
            X_mb = tf.multiply(X_mb, mask_slice) #masking
            
            H_mb = embedder_model(X_mb) #recall
            H_mb = tf.multiply(H_mb, mask_slice) #masking
            E_hat_mb = generator_model(Z_mb) #recall
            E_hat_mb = tf.multiply(E_hat_mb, mask_slice) #masking
            H_hat_mb = supervisor_model(E_hat_mb) #recall
            H_hat_mb = tf.multiply(H_hat_mb, mask_slice) #masking
            
            # Synthetic data
            X_hat_mb = recovery_model(H_hat_mb)
            X_hat_mb = tf.multiply(X_hat_mb, mask_slice) #masking
            
            # Discriminator
            Y_fake_mb = discriminator_model(H_hat_mb)
            Y_fake_mb = tf.multiply(Y_fake_mb, mask_slice) #masking
            Y_real_mb = discriminator_model(H_mb)
            Y_real_mb = tf.multiply(Y_real_mb, mask_slice) #masking
            Y_fake_e_mb = discriminator_model(E_hat_mb)
            Y_fake_e_mb = tf.multiply(Y_fake_e_mb, mask_slice) #masking

            # Check discriminator loss before updating
            disc_loss = get_discriminator_loss(Y_real_mb, Y_fake_mb, Y_fake_e_mb, mask_slice)
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
        
        for itt in range(iterations):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb = np.nan_to_num(X_mb, nan=-1) #fill in NaNs with -1 (as the model can't take NaNs)

            #X_mb[0, 0:2, :] = 999 #masking rows 3 to 6
            #X_mb[1, 1:3, :] = 999 #other variations of masking
            #X_mb[1, 5, :2] = 999 
            
            # Train embedder
            step_e_loss = train_step_embedder(X_mb)
           
            # Checkpoint
            if itt % 1 == 0:
                print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )

        print('Finish Embedding Network Training')
        
        #2. Training only with supervised loss
        print('Start Training with Supervised Loss Only')

        for itt in range(iterations):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb = np.nan_to_num(X_mb, nan=-1)

            #X_mb[0, 0:2, :] = 999 #masking rows 3 to 6
            #X_mb[1, 1:3, :] = 999 #other variations of masking
            #X_mb[1, 5, :2] = 999

            # Random vector generation 
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            print(X_mb.shape)
            print(Z_mb.shape)
            # Train generator
            step_gen_s_loss = train_step_generator_s(X_mb, Z_mb)

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
                X_mb = np.nan_to_num(X_mb, nan=-1)
                #X_mb[0, 0:2, :] = 999 #masking rows 3 to 6
                #X_mb[1, 1:3, :] = 999 #other variations of masking
                #X_mb[1, 5, :] = 999

                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                # Train generator and embedder
                emb_T0_loss, emb_loss, g_loss_u, gen_s_loss, g_loss_v = train_step_joint(X_mb, Z_mb)
            
            # Discriminator training        
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            X_mb = np.nan_to_num(X_mb, nan=-1)           
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

from data_loading import real_data_loading, sine_data_generation

data_name = 'prism'
seq_len = 50
#if we have short seq length and there are no (or a few?) unmasked rows (the real data), 
# the loss might eventually become nan!

if data_name in ['stock', 'energy', 'prism']:
 ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
  # Set number of samples and its dimensions
  no, dim = 15, 5
  ori_data = sine_data_generation(no, seq_len, dim)
    
print(data_name + ' dataset is ready.')

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 4
parameters['num_layer'] = 3
parameters['iterations'] = 2
parameters['batch_size'] = 2

generated_data = timegan(ori_data, parameters)
print('Finish Synthetic Data Generation')
print(generated_data)

