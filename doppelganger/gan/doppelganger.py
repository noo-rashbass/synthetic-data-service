import tensorflow as tf
import numpy as np
from tqdm import tqdm
import datetime
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DoppelGANger(tf.keras.Model):
    def __init__(self, epoch, batch_size, 
                 data_feature, data_attribute, real_attribute_mask, data_gen_flag,
                 seq_len, data_feature_outputs, data_attribute_outputs,
                 generator, discriminator, 
                 d_rounds, g_rounds, d_gp_coe,
                 num_packing,
                 attr_discriminator=None,
                 attr_d_gp_coe=None, g_attr_d_coe=None,
                 attribute_latent_dim=5, feature_latent_dim=5,
                 fix_feature_network=False,
                 g_lr=0.001, g_beta1=0.5,
                 d_lr=0.001, d_beta1=0.5,
                 attr_d_lr=0.001, attr_d_beta1=0.5,
                 checkpoints=None,
                 cumsum=False):
        super(DoppelGANger, self).__init__()
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_feature = data_feature
        self.data_attribute = data_attribute
        self.real_attribute_mask = real_attribute_mask
        self.data_gen_flag = data_gen_flag
        self.seq_len = seq_len
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.generator = generator
        self.discriminator = discriminator
        self.attr_discriminator = attr_discriminator
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.d_gp_coe = d_gp_coe
        self.attr_d_gp_coe = attr_d_gp_coe
        self.g_attr_d_coe = g_attr_d_coe
        self.attribute_latent_dim = attribute_latent_dim
        self.feature_latent_dim = feature_latent_dim
        self.fix_feature_network = fix_feature_network
        self.num_packing = num_packing
        self.g_lr = g_lr
        self.g_beta1 = g_beta1
        self.d_lr = d_lr
        self.d_beta1 = d_beta1
        self.attr_d_lr = attr_d_lr
        self.attr_d_beta1 = attr_d_beta1
        self.cumsum = cumsum
        self.checkpoints = checkpoints

        # optimizers
        self.g_op = tf.keras.optimizers.Adam(self.g_lr, self.g_beta1)
        self.d_op = tf.keras.optimizers.Adam(self.d_lr, self.d_beta1)
        if self.attr_discriminator is not None:
            self.attr_d_op = tf.keras.optimizers.Adam(self.attr_d_lr, self.attr_d_beta1)
        
        # checkpoints
        self.d_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.d_op, net=self.discriminator)
        self.d_manager = tf.train.CheckpointManager(self.d_ckpt, "tf_ckpts_d", max_to_keep=3)
        self.ad_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.attr_d_op, net=self.attr_discriminator)
        self.ad_manager = tf.train.CheckpointManager(self.ad_ckpt, "tf_ckpts_ad", max_to_keep=3)
        self.g_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.g_op, net=self.generator)
        self.g_manager = tf.train.CheckpointManager(self.g_ckpt, "tf_ckpts_g", max_to_keep=3)

        self.check_data()

        if self.data_feature.shape[1] % self.seq_len != 0:
            raise Exception("length must be a multiple of sample_len")
        self.sample_time = int(self.data_feature.shape[1] / self.seq_len)
        self.sample_feature_dim = self.data_feature.shape[2]
        self.sample_attribute_dim = self.data_attribute.shape[1]
        self.sample_real_attribute_dim = 0
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.sample_real_attribute_dim += self.data_attribute_outputs[i].dim

        self.EPS = 1e-8

    def check_data(self):
        """
        checks if input data is in the correct format
        """
        self.gen_flag_dims = []

        dim = 0
        for output in self.data_feature_outputs:
            if output.is_gen_flag:
                if output.dim != 2:
                    raise Exception("gen flag output's dim should be 2")
                self.gen_flag_dims = [dim, dim + 1]
                break
            dim += output.dim
        if len(self.gen_flag_dims) == 0:
            raise Exception("gen flag not found")

        if (self.data_feature.shape[2] != np.sum([t.dim for t in self.data_feature_outputs])):
            raise Exception(
                "feature dimension does not match data_feature_outputs")

        if len(self.data_gen_flag.shape) != 2:
            raise Exception("data_gen_flag should be 2 dimension")

        self.data_gen_flag = np.expand_dims(self.data_gen_flag, 2)
        
    # this is redundant now   
    def compile(self):
        super(DoppelGANger, self).compile()
        self.g_op = tf.keras.optimizers.Adam(self.g_lr, self.g_beta1)
        self.d_op = tf.keras.optimizers.Adam(self.d_lr, self.d_beta1)
        if self.attr_discriminator is not None:
            self.attr_d_op = tf.keras.optimizers.Adam(self.attr_d_lr, self.attr_d_beta1)

    def gen_attribute_input_noise(self, num_sample):
        return np.random.normal(size=[num_sample, self.attribute_latent_dim])
    
    def gen_feature_input_noise(self, num_sample, length):
        return np.random.normal(size=[num_sample, length, self.feature_latent_dim])
    
    def gen_feature_input_data_free(self, num_sample):
        return np.zeros([num_sample, self.seq_len * self.sample_feature_dim], dtype=np.float32)

    def sample_from(self, real_attribute_input_noise, addi_attribute_input_noise, feature_input_noise,
                    feature_input_data, given_attribute=None, return_gen_flag_feature=False):
        features = []
        attributes = []
        gen_flags = []
        lengths = []
        round_ = int(math.ceil(float(feature_input_noise.shape[0]) / self.batch_size))
        print(round_)

        for i in range(round_):
            # generate as usual (without specifying attribute)
            if given_attribute is None:
                if feature_input_data.ndim == 2:
                    (sub_features, sub_attributes, sub_gen_flags, sub_lengths, _) = \
                        self.generator(
                            real_attribute_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                            addi_attribute_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                            feature_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                            feature_input_data[i * self.batch_size : (i + 1) * self.batch_size], #dim(None, seq_len*sample_feature_dim)
                            train=False)

                else:
                    (sub_features, sub_attributes, sub_gen_flags, sub_lengths, _) = \
                        self.generator(
                            real_attribute_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                            addi_attribute_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                            feature_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                            feature_input_data[i * self.batch_size : (i + 1) * self.batch_size], #dim(None, None, seq_len*sample_feature_dim)
                            train=False)
            # if you want to generate based on a given attribute                
            else:
                (sub_features, sub_attributes, sub_gen_flags, sub_lengths, _) = \
                    self.generator(
                        None,
                        addi_attribute_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                        feature_input_noise[i * self.batch_size : (i + 1) * self.batch_size],
                        feature_input_data[i * self.batch_size : (i + 1) * self.batch_size],
                        train=False,
                        attribute=given_attribute[i * self.batch_size : (i + 1) * self.batch_size])
            
            features.append(sub_features)
            attributes.append(sub_attributes)
            gen_flags.append(sub_gen_flags)
            lengths.append(sub_lengths)
        
        features = np.concatenate(features, axis=0)
        attributes = np.concatenate(attributes, axis=0)
        gen_flags = np.concatenate(gen_flags, axis=0)
        lengths = np.concatenate(lengths, axis=0)

        if not return_gen_flag_feature:
            features = np.delete(features, self.gen_flag_dims, axis=2)

        assert len(gen_flags.shape) == 3
        assert gen_flags.shape[2] == 1
        gen_flags = gen_flags[:, :, 0]

        return features, attributes, gen_flags, lengths 

    def gen_loss(self, d_fake_train_tf, attr_d_fake_train_tf):

        """
        calculates generator loss
        Args:
        d_fake_train_tf: discriminator result on fake data
        attr_d_fake_train_tf: discriminator result on fake data attribute

        Returns:
        g_loss: generator loss
        """

        #batch_size = tf.shape(self.batch_feature_input_noise[0])[0] #check
        batch_size = self.batch_size

        #1. Generator loss
        g_loss_d = -tf.reduce_mean(d_fake_train_tf)
        if self.attr_discriminator is not None:
            g_loss_attr_d = -tf.reduce_mean(attr_d_fake_train_tf)
            g_loss = (g_loss_d + self.g_attr_d_coe * g_loss_attr_d)
        else:
            g_loss = g_loss_d

        return g_loss

    def disc_loss(self, d_fake_train_tf, d_real_train_tf, g_output_feature_train_tf, g_output_attribute_train_tf,
                    batch_data_feature, batch_data_attribute):
        """
        calculates discriminator loss
        Args:
        d_fake_train_tf: discriminator result on fake data
        attr_d_fake_train_tf: discriminator result on fake data attribute
        g_output_feature_train_tf: generated features output from generator
        g_output_attribute_train_tf: generated attributes output from generator
        batch_data_feature: features from original dataset
        batch_data_attribute: attributes from original dataset

        Returns:
        disc_loss: discriminator loss
        """
        
        #batch_size = tf.shape(self.batch_feature_input_noise[0])[0] 
        batch_size = self.batch_size #check

        #2. Discriminator loss
        d_loss_fake = tf.reduce_mean(d_fake_train_tf) 
        d_loss_real = -tf.reduce_mean(d_real_train_tf)

        #interpolate data for gp
        alpha_dim2 = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
        alpha_dim3 = tf.expand_dims(alpha_dim2, 2)
        differences_input_feature = (g_output_feature_train_tf - batch_data_feature)
        interpolates_input_feature = (batch_data_feature + alpha_dim3 * differences_input_feature)
        differences_input_attribute = (g_output_attribute_train_tf - batch_data_attribute)
        interpolates_input_attribute = (batch_data_attribute + (alpha_dim2 * differences_input_attribute))
        
        with tf.GradientTape() as tape:
            tape.watch([interpolates_input_feature, interpolates_input_attribute])
            predict = self.discriminator([interpolates_input_feature, interpolates_input_attribute])

        # Calculate the gradients w.r.t to this interpolated patient
        grad = tape.gradient(predict, [interpolates_input_feature, interpolates_input_attribute])

        slopes1 = tf.math.reduce_sum(tf.math.square(grad[0]),axis=[1, 2])
        slopes2 = tf.math.reduce_sum(tf.math.square(grad[1]), axis=[1])
        slopes = tf.math.sqrt(slopes1 + slopes2 + self.EPS)
        d_loss_gp = tf.math.reduce_mean((slopes - 1.)**2)

        ################ CUM SUM   ###############################################################
        # This is an attempt to capture bimodal distribution in data columns using cumulative distribution
        # not sure if it actually works
        if self.cumsum:
            batch_data_feature_cum = tf.cast(tf.cumsum(batch_data_feature, axis=1), tf.float32)
            g_output_feature_train_tf_cum = tf.cumsum(g_output_feature_train_tf, axis=1)
            batch_data_attribute_cum = tf.cast(tf.cumsum(batch_data_attribute, axis=1), tf.float32)
            g_output_attribute_train_tf_cum = tf.cumsum(g_output_attribute_train_tf, axis=1)

            alpha_dim4 = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            alpha_dim5 = tf.expand_dims(alpha_dim2, 2)
            differences_input_feature_cum = (g_output_feature_train_tf_cum - batch_data_feature_cum)
            interpolates_input_feature_cum = (batch_data_feature_cum + alpha_dim5 * differences_input_feature_cum)
            differences_input_attribute_cum = (g_output_attribute_train_tf_cum - batch_data_attribute_cum)
            interpolates_input_attribute_cum = (batch_data_attribute_cum + (alpha_dim4 * differences_input_attribute_cum))
            
            #not sure to use tf.grad or tf.gradtape
            with tf.GradientTape() as tape2:
                tape2.watch([interpolates_input_feature_cum, interpolates_input_attribute_cum])
                predict_cum = self.discriminator([interpolates_input_feature_cum, interpolates_input_attribute_cum])

            # Calculate the gradients w.r.t to this interpolated patient
            grad_cum = tape2.gradient(predict_cum, [interpolates_input_feature_cum, interpolates_input_attribute_cum])

            slopes1_cum = tf.math.reduce_sum(tf.math.square(grad_cum[0]),axis=[1, 2])
            slopes2_cum = tf.math.reduce_sum(tf.math.square(grad_cum[1]), axis=[1])
            slopes_cum = tf.math.sqrt(slopes1_cum + slopes2_cum + self.EPS)
            d_loss_gp_cum = tf.math.reduce_mean((slopes_cum - 1.)**2)
        else:
            d_loss_gp_cum = 0

        ############### END OF CUM SUM   ############################################################
        
        d_loss = (d_loss_fake + d_loss_real + self.d_gp_coe * (d_loss_gp + d_loss_gp_cum))
    
        return d_loss 

    def attr_d_loss(self, attr_d_fake_train_tf, attr_d_real_train_tf, g_output_attribute_train_tf,
                    batch_data_attribute):

        """
        calculates attribute discriminator loss
        Args:
        attr_d_fake_train_tf: discriminator result on fake data attribute
        attr_d_real_train_tf: dicriminator result on real data attribute
        g_output_attribute_train_tf: generated attributes output from generator
        batch_data_attribute: attributes from original dataset

        Returns:
        attr_d_loss: attribute discriminator loss
        """

        #batch_size = tf.shape(self.batch_feature_input_noise[0])[0] 
        batch_size = self.batch_size #check

        #Attr Discriminator loss
        if self.attr_discriminator is not None:
            attr_d_loss_fake = tf.math.reduce_mean(attr_d_fake_train_tf)
            attr_d_loss_real = -tf.math.reduce_mean(attr_d_real_train_tf)
            alpha_dim2 = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            differences_input_attribute = (g_output_attribute_train_tf - batch_data_attribute)
            interpolates_input_attribute = (batch_data_attribute + (alpha_dim2 * differences_input_attribute))

            with tf.GradientTape() as tape:
                tape.watch(interpolates_input_attribute)
                predict = self.attr_discriminator(interpolates_input_attribute)

            # Calculate the gradients w.r.t to this interpolated patient
            grad = tape.gradient(predict, [interpolates_input_attribute])

            slopes1 = tf.math.reduce_sum(tf.math.square(grad[0]), axis=[1])
            slopes = tf.math.sqrt(slopes1 + self.EPS)
            attr_d_loss_gp = tf.reduce_mean((slopes - 1.)**2)

            ##### CUM SUM #################################################
            # This is an attempt to capture bimodal distribution in data columns using cumulative distribution
            # not sure if it actually works
            if self.cumsum:
                batch_data_attribute_cum = tf.cast(tf.cumsum(batch_data_attribute, axis=1), tf.float32)
                g_output_attribute_train_tf_cum = tf.cumsum(g_output_attribute_train_tf, axis=1)

                alpha_dim4 = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
                differences_input_attribute_cum = (g_output_attribute_train_tf_cum - batch_data_attribute_cum)
                interpolates_input_attribute_cum = (batch_data_attribute_cum + (alpha_dim4 * differences_input_attribute_cum))

                #not sure to use tf.grad or tf.gradtape
                with tf.GradientTape() as tape2:
                    tape2.watch(interpolates_input_attribute_cum)
                    predict_cum = self.attr_discriminator(interpolates_input_attribute_cum)

                # Calculate the gradients w.r.t to this interpolated patient
                grad_cum = tape2.gradient(predict_cum, [interpolates_input_attribute_cum])

                slopes2_cum = tf.math.reduce_sum(tf.math.square(grad_cum[0]), axis=[1])
                slopes_cum = tf.math.sqrt(slopes2_cum + self.EPS)
                attr_d_loss_gp_cum = tf.math.reduce_mean((slopes_cum - 1.)**2)
            else:
                attr_d_loss_gp_cum = 0
            
            ###### END OF CUM SUM ####################################################

            attr_d_loss = (attr_d_loss_fake + attr_d_loss_real + self.attr_d_gp_coe * (attr_d_loss_gp + attr_d_loss_gp_cum))
        
        return attr_d_loss 

    def train_step_d(self, batch_real_attribute_input_noise, batch_addi_attribute_input_noise,
                        batch_feature_input_noise, batch_feature_input_data,
                        batch_data_feature, batch_data_attribute): 
        """
        trains the discriminator model
        Args:
        batch_real_attribute_input_noise: random noise for attributes in dataset
        batch_addi_attribute_input_noise: random noise for additional attributes
        batch_feature_input_noise: random noise for additional features
        batch_feature_input_data: random noise for features in dataset
        batch_data_feature: features from original dataset
        batch_data_attribute: attributes from original dataset

        Returns:
        d_loss: discriminator loss

        """
        
        with tf.GradientTape() as tape:

            (g_output_feature_train_tf, g_output_attribute_train_tf, 
            g_output_gen_flag_train_tf, g_output_length_train_tf, g_output_argmax_train_tf) = \
            self.generator( batch_real_attribute_input_noise,
                                    batch_addi_attribute_input_noise,
                                    batch_feature_input_noise,
                                    batch_feature_input_data,
                                    train=True)

            if self.fix_feature_network:
                g_output_feature_train_tf = tf.zeros_like(g_output_feature_train_tf)
                g_output_gen_flag_train_tf = tf.zeros_like(g_output_gen_flag_train_tf)
                g_output_attribute_train_tf *= self.real_attribute_mask_tensor

            d_fake_train_tf = self.discriminator([g_output_feature_train_tf, g_output_attribute_train_tf]) #should i call fit?
            d_real_train_tf = self.discriminator([batch_data_feature, batch_data_attribute]) #??? 
            d_loss = self.disc_loss(d_fake_train_tf, d_real_train_tf, g_output_feature_train_tf, 
                                    g_output_attribute_train_tf, batch_data_feature, batch_data_attribute)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_op.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        return d_loss

    def train_step_attr_d(self, batch_real_attribute_input_noise, batch_addi_attribute_input_noise,
                          batch_feature_input_noise, batch_feature_input_data,
                          batch_data_feature, batch_data_attribute): 
        
        """
        trains the attribute discriminator model
        Args:
        batch_real_attribute_input_noise: random noise for attributes in dataset
        batch_addi_attribute_input_noise: random noise for additional attributes
        batch_feature_input_noise: random noise for additional features
        batch_feature_input_data: random noise for features in dataset
        batch_data_feature: features from original dataset
        batch_data_attribute: attributes from original dataset

        Returns:
        ad_loss: attribute discriminator loss

        """

        with tf.GradientTape() as tape:

            (g_output_feature_train_tf, g_output_attribute_train_tf, 
            g_output_gen_flag_train_tf, g_output_length_train_tf, g_output_argmax_train_tf) = \
            self.generator( batch_real_attribute_input_noise,
                                    batch_addi_attribute_input_noise,
                                    batch_feature_input_noise,
                                    batch_feature_input_data,
                                    train=True)

            if self.fix_feature_network:
                g_output_feature_train_tf = tf.zeros_like(g_output_feature_train_tf)
                g_output_gen_flag_train_tf = tf.zeros_like(g_output_gen_flag_train_tf)
                g_output_attribute_train_tf *= self.real_attribute_mask_tensor

            attr_d_fake_train_tf = self.attr_discriminator(g_output_attribute_train_tf)
            attr_d_real_train_tf = self.attr_discriminator(batch_data_attribute) #in place of real_attr_pl
            ad_loss = self.attr_d_loss(attr_d_fake_train_tf, attr_d_real_train_tf, 
                                        g_output_attribute_train_tf,batch_data_attribute)
        grads = tape.gradient(ad_loss, self.attr_discriminator.trainable_weights)
        self.attr_d_op.apply_gradients(zip(grads, self.attr_discriminator.trainable_weights))

        return ad_loss
    
    def train_step_gen(self, batch_real_attribute_input_noise, batch_addi_attribute_input_noise, 
                        batch_feature_input_noise, batch_feature_input_data, attribute=None):

        """
        trains the generator model
        Args:
        batch_real_attribute_input_noise: random noise for attributes in dataset
        batch_addi_attribute_input_noise: random noise for additional attributes
        batch_feature_input_noise: random noise for additional features
        batch_feature_input_data: random noise for features in dataset

        Returns:
        d_loss: generator loss

        """

        with tf.GradientTape() as tape:
            
            (g_output_feature_train_tf, g_output_attribute_train_tf, 
            g_output_gen_flag_train_tf, g_output_length_train_tf, g_output_argmax_train_tf) = \
            self.generator( batch_real_attribute_input_noise,
                                    batch_addi_attribute_input_noise,
                                    batch_feature_input_noise,
                                    batch_feature_input_data,
                                    train=True)

            if self.fix_feature_network:
                g_output_feature_train_tf = tf.zeros_like(g_output_feature_train_tf)
                g_output_gen_flag_train_tf = tf.zeros_like(g_output_gen_flag_train_tf)
                g_output_attribute_train_tf *= self.real_attribute_mask_tensor
            
            d_fake_train_tf = self.discriminator([g_output_feature_train_tf, g_output_attribute_train_tf])
            attr_d_fake_train_tf = self.attr_discriminator(g_output_attribute_train_tf)
            g_loss = self.gen_loss(d_fake_train_tf, attr_d_fake_train_tf)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_op.apply_gradients(zip(grads, self.generator.trainable_weights))

        return g_loss
    
    def train_step(self):

        """
        trains the doppelGANger algorithm for self.epochs number of times
        """

        # calling fit does not work bcoz of the way the last iteration in each epoch is handled
        # num_features = self.data_feature.shape[2] 
        # separate the features and attributes
        # data_feature_re = tf.slice(data_all_in, [0,0,0], [-1,-1,num_features]) 
        # print(tf.slice(data_all_in, [0,0,num_features], [-1, 1, -1]))
        # data_attribute_re = tf.squeeze(tf.slice(data_all_in, [0,0,num_features], [-1, 1, -1]), axis=1)

        # definitely not resetting but not sure if it is reloading properly
        if self.checkpoints == None:
            #restore from latest checkpoint
            self.g_ckpt.restore(self.g_manager.latest_checkpoint)
            if self.g_manager.latest_checkpoint:
                print("G restored from {}".format(self.g_manager.latest_checkpoint))
            else:
                print("G initializing from scratch.")

            self.d_ckpt.restore(self.d_manager.latest_checkpoint)
            if self.d_manager.latest_checkpoint:
                print("D restored from {}".format(self.d_manager.latest_checkpoint))
            else:
                print("D initializing from scratch.")

            self.ad_ckpt.restore(self.ad_manager.latest_checkpoint)
            if self.ad_manager.latest_checkpoint:
                print("AD restored from {}".format(self.ad_manager.latest_checkpoint))
            else:
                print("AD initializing from scratch.")
        else:
            #restore from specified checkpoints
            self.g_ckpt.restore(checkpoints[0])
            self.d_ckpt.restore(checkpoints[1])
            self.ad_ckpt.restore(checkpoints[2])
            print("G restored from {}".format(checkpoints[0]))
            print("D restored from {}".format(checkpoints[1]))
        
        for i in range(self.epoch):
            print("epoch: ", i)
            batch_num = self.data_feature.shape[0] // self.batch_size

            data_id = np.random.choice(self.data_feature.shape[0], size=(self.data_feature.shape[0], self.num_packing))

            for batch_id in range(batch_num):
                batch_data_id = data_id[batch_id * self.batch_size: (batch_id + 1) * self.batch_size, 0]
                batch_data_feature = self.data_feature[batch_data_id] 
                batch_data_attribute = self.data_attribute[batch_data_id]

                batch_real_attribute_input_noise = self.gen_attribute_input_noise(self.batch_size) 
                batch_addi_attribute_input_noise = self.gen_attribute_input_noise(self.batch_size) 
                batch_feature_input_noise = self.gen_feature_input_noise(self.batch_size, self.sample_time) 
                batch_feature_input_data = self.gen_feature_input_data_free(self.batch_size) 

                # train discriminator model
                for _ in range(self.d_rounds): #d_rounds-1 in ori code?
                    d_loss = self.train_step_d(batch_real_attribute_input_noise, 
                                                    batch_addi_attribute_input_noise,
                                                    batch_feature_input_noise,
                                                    batch_feature_input_data,
                                                    batch_data_feature,       
                                                    batch_data_attribute) 

                    if self.attr_discriminator is not None:
                        ad_loss = self.train_step_attr_d(batch_real_attribute_input_noise,
                                                    batch_addi_attribute_input_noise,
                                                    batch_feature_input_noise,
                                                    batch_feature_input_data,
                                                    batch_data_feature,   
                                                    batch_data_attribute)

                # trains attribute discriminator model
                if self.attr_discriminator is not None:
                    ad_loss = self.train_step_attr_d(batch_real_attribute_input_noise,
                                                    batch_addi_attribute_input_noise,
                                                    batch_feature_input_noise,
                                                    batch_feature_input_data,
                                                    batch_data_feature,       
                                                    batch_data_attribute)

                # trains generator model
                for _ in range(self.g_rounds):
                    g_loss = self.train_step_gen(batch_real_attribute_input_noise,
                                                    batch_addi_attribute_input_noise,
                                                    batch_feature_input_noise,
                                                    batch_feature_input_data)
            
            self.d_ckpt.step.assign_add(1)
            self.g_ckpt.step.assign_add(1)
            self.ad_ckpt.step.assign_add(1)
            save_path_g = self.g_manager.save()
            save_path_d = self.d_manager.save()
            save_path_ad = self.ad_manager.save()

            print("d_loss: ", d_loss.numpy(), ", ad_loss: ", ad_loss.numpy(), ", g_loss: ", g_loss.numpy())
            
        return {"d_loss": d_loss, "ad_loss": ad_loss, "g_loss": g_loss}



# from load_data import *
# from util import *
# seq_len = 130
# batch_size = 64
# epochs = 2
# (data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = load_data("data")

# print("-----DATA LOADING PART-----")
# print(data_feature.shape)          # original features_dim         
# print(data_attribute.shape)        # original attributes_dim
# print(data_gen_flag.shape)
# num_real_attribute = len(data_attribute_outputs)

# (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = \
#     normalize_per_sample(data_feature, data_attribute, data_feature_outputs,data_attribute_outputs)

# print("-----DATA NORMALIZATION PART-----")
# print(real_attribute_mask)
# print(data_feature.shape)
# attributes_dim = data_attribute.shape[1]    # attributes_dim to be fed into model
# print(data_attribute.shape)       
# print(len(data_attribute_outputs))

# print("-----ADD GEN FLAG PART -----")
# data_feature, data_feature_outputs = add_gen_flag(
#         data_feature, data_gen_flag, data_feature_outputs, seq_len)
# features_dim = data_feature.shape[2]    # features dim to be fed into model
# print(data_feature.shape)        
# print(len(data_feature_outputs))

# from network import make_discriminator, make_attrdiscriminator
# from networkGenerator import DoppelGANgerGenerator

# discriminator_model = make_discriminator(seq_len, features_dim, attributes_dim)
# attrdiscriminator_model = make_attrdiscriminator(attributes_dim)
# discriminator_model.summary()
# attrdiscriminator_model.summary()

# generator = DoppelGANgerGenerator(
#         feed_back=False,
#         noise=True,
#         feature_outputs=data_feature_outputs,
#         attribute_outputs=data_attribute_outputs,
#         real_attribute_mask=real_attribute_mask,
#         sample_len=seq_len)


# gan = DoppelGANger(
#     epoch=epochs, 
#     batch_size=batch_size, 
#     data_feature=data_feature, 
#     data_attribute=data_attribute, 
#     real_attribute_mask=real_attribute_mask, 
#     data_gen_flag=data_gen_flag,
#     seq_len=seq_len, 
#     data_feature_outputs=data_feature_outputs, 
#     data_attribute_outputs=data_feature_outputs,
#     generator = generator, 
#     discriminator = discriminator_model, 
#     d_rounds=1, 
#     g_rounds=1, 
#     d_gp_coe=10.,
#     num_packing=1,
#     attr_discriminator=attrdiscriminator_model,
#     attr_d_gp_coe=10., 
#     g_attr_d_coe=1.0)


# #gan.train_step()
# gan.compile()

# #combine data attributes and features into one to be fed into the model
# # data_attribute_in = tf.expand_dims(data_attribute, axis=1)
# # data_attribute_in = tf.repeat(data_attribute_in, repeats=seq_len, axis=1)
# # data_all_in = tf.cast(tf.concat([data_feature, data_attribute_in], axis=2), dtype=tf.float32)

# data_all_in = np.ones((1,1))

# print("----TRAINING-----")

# gan.train_step()


# print("----FINISHED TRAINING-----")

# print("----START GENERATING------")
# total_generate_num_sample = 1347

# if data_feature.shape[1] % seq_len != 0:
#     raise Exception("length must be a multiple of sample_len")
# length = int(data_feature.shape[1] / seq_len)
# real_attribute_input_noise = gan.gen_attribute_input_noise(total_generate_num_sample) #(?,5)
# addi_attribute_input_noise = gan.gen_attribute_input_noise(total_generate_num_sample) #(?,5)
# feature_input_noise = gan.gen_feature_input_noise(total_generate_num_sample, length) #(?,1,5)
# input_data = gan.gen_feature_input_data_free(total_generate_num_sample) #(?,28)

# features, attributes, gen_flags, lengths = \
#     gan.sample_from(real_attribute_input_noise, addi_attribute_input_noise,feature_input_noise, input_data)
# # specify given_attribute parameter, if you want to generate
# # data according to an attribute
# print("----SAMPLE FROM PART-----")
# print(features.shape)
# print(attributes.shape)
# print(gen_flags.shape)
# print(lengths.shape)

# features, attributes = renormalize_per_sample(features, attributes, data_feature_outputs,
#     data_attribute_outputs, gen_flags, num_real_attribute=num_real_attribute)
# print("----RENORMALIZATION PART-----")
# print(features.shape)
# print(attributes.shape)

# np.savez(
#         "generated_data_train.npz",
#         data_feature=features,
#         data_attribute=attributes,
#         data_gen_flag=gen_flags)

# print("Done")