import tensorflow as tf
#from op import linear, batch_norm, flatten
from tensorflow.keras import layers
from output import OutputType, Normalization, Output
from util import *
from load_data import *
import numpy as np
from enum import Enum
import os

import sys
import output
sys.modules["output"] = output
#from data_loading import sine_data_generation_f_a, real_data_loading_prism, renormalize
#tf.keras.backend.set_floatx('float32')

class RNNInitialStateType(Enum):
    ZERO = "ZERO"
    RANDOM = "RANDOM"
    VARIABLE = "VARIABLE"

class DoppelGANgerGeneratorRNN(tf.keras.layers.Layer):
    def __init__(self, feature_outputs, sample_len, noise,
                feature_num_layers=1, feature_num_units=100, 
                initial_state=RNNInitialStateType.RANDOM,*args, **kwargs):
        super(DoppelGANgerGeneratorRNN, self).__init__(*args, **kwargs)
        
        
        self.feature_outputs = feature_outputs
        self.sample_len = sample_len
        self.feature_num_layers = feature_num_layers
        self.feature_num_units =  feature_num_units
        self.initial_state = initial_state            
        
        self.noise = noise

        self.feature_out_dim = (np.sum([t.dim for t in feature_outputs]) *
                                self.sample_len)
    
        self.outputs ={}
        self.outputs ={}

        for i in range(self.sample_len):
            for k, output in enumerate(self.feature_outputs):
                self.outputs[str(i) + ":" + str(k)] = tf.keras.layers.Dense(output.dim)
        
        self.gen_flag_id = None
        for i in range(len(self.feature_outputs)):
            if self.feature_outputs[i].is_gen_flag:
                self.gen_flag_id = i
                break
        if self.gen_flag_id is None:
            raise Exception("cannot find gen_flag_id")
        if self.feature_outputs[self.gen_flag_id].dim != 2:
            raise Exception("gen flag output's dim should be 2")
        
        #self.rnn_network = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(self.feature_num_units) for _ in range(self.feature_num_layers)]), return_state=True)

        self.rnn_network = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(self.feature_num_units), return_state=True)

    def call(self, all_discrete_attribute, feature_input_noise, feature_input_data):

        feature_input_data = feature_input_data
        feature_input_data_dim = \
                    len(tf.convert_to_tensor(feature_input_data).get_shape().as_list())
        if feature_input_data_dim == 3:
            feature_input_data_reshape = tf.transpose(
                       feature_input_data, [1, 0, 2])

        feature_input_noise_reshape = tf.transpose(
            feature_input_noise, [1, 0, 2])

        time = tf.convert_to_tensor(feature_input_noise).get_shape().as_list()[1]
        
        if time is None:
            time = tf.shape(feature_input_noise)[1]      
        batch_size = tf.shape(feature_input_noise)[0]

        initial_all_output = tf.TensorArray(tf.float32, time)
        initial_gen_flag = tf.ones((batch_size, 1))
        initial_all_gen_flag = tf.TensorArray(tf.float32, time * self.sample_len)
        initial_all_cur_argmax = tf.TensorArray(tf.int64, time * self.sample_len)
        initial_last_cell_output = tf.zeros((batch_size, self.feature_num_units))
        if feature_input_data_dim == 2:
            initial_last_output=feature_input_data 
            
        else:
            initial_last_output = feature_input_data_reshape[0]


        
        if self.initial_state == RNNInitialStateType.ZERO:
            initial_state_ = rnn_network.zero_state(
                batch_size, tf.float32)
        elif self.initial_state == RNNInitialStateType.RANDOM:
            # For GRU
            # input_ = tf.random.normal([batch_size, self.feature_num_units, 1]) ### Not sure if i got this right
            # _, initial_state_ = self.rnn_network(input_)

            # For LSTM
            input_ = tf.random.normal([batch_size, self.feature_num_units, 1]) ### Not sure if i got this right
            _, initial_state_h, intial_state_c = self.rnn_network(input_)
            initial_state_ = [initial_state_h, intial_state_c]

        else:
            raise NotImplementedError

        def compute(i, state, last_output, all_output,
                            gen_flag, all_gen_flag, all_cur_argmax,
                            last_cell_output):

            
            input_all = [tf.cast(all_discrete_attribute, dtype=tf.float32)]            
            
            if self.noise:
                input_all.append(tf.cast(feature_input_noise_reshape[i], dtype=tf.float32))
                
            input_all = tf.concat(input_all, axis=1)

            input_all = tf.expand_dims(input_all, axis=2)

            # For GRU
            # cell_new_output, new_state = self.rnn_network(input_all, initial_state=state)

            cell_new_output, new_state_h, new_state_c = self.rnn_network(input_all, initial_state=state)
            new_state = [new_state_h, new_state_c]


            new_output_all = []
            id_ = 0
            for j in range(self.sample_len):
                for k in range(len(self.feature_outputs)):
                    output = self.feature_outputs[k]
                    sub_output = self.outputs[str(j) + ":" + str(k)](cell_new_output)
                    if (output.type_ == OutputType.DISCRETE):
                        sub_output = tf.nn.softmax(sub_output)
                    elif (output.type_ == OutputType.CONTINUOUS):
                        if (output.normalization ==
                                Normalization.ZERO_ONE):
                            sub_output = tf.nn.sigmoid(sub_output)
                        elif (output.normalization ==
                                Normalization.MINUSONE_ONE):
                            sub_output = tf.nn.tanh(sub_output)
                        else:
                            raise Exception("unknown normalization"
                                            " type")
                    else:
                        raise Exception("unknown output type")
                    new_output_all.append(sub_output)
                    id_ += 1
            new_output = tf.concat(new_output_all, axis=1)

            
            for j in range(self.sample_len):
                all_gen_flag = all_gen_flag.write(
                    i * self.sample_len + j, gen_flag)
                cur_gen_flag = tf.cast(tf.equal(tf.argmax(
                    new_output_all[(j * len(self.feature_outputs) +
                                    self.gen_flag_id)],
                    axis=1), 0), dtype=tf.float32) 
                cur_gen_flag = tf.reshape(cur_gen_flag, [-1, 1])
                all_cur_argmax = all_cur_argmax.write(
                    i * self.sample_len + j,
                    tf.argmax(
                        new_output_all[(j * len(self.feature_outputs) +
                                        self.gen_flag_id)],
                        axis=1))
                
                gen_flag = gen_flag * cur_gen_flag

            return (i + 1, 
                    new_state,
                    new_output, 
                    all_output.write(i, new_output),
                    gen_flag,
                    all_gen_flag,
                    all_cur_argmax,
                    cell_new_output) 

        (i, state, _, feature, _, gen_flag, cur_argmax,
            cell_output) = \
            tf.while_loop(
                lambda a, b, c, d, e, f, g, h:
                tf.logical_and(a < time,
                                tf.equal(tf.reduce_max(e), 1)),
                compute,
                (tf.cast(0, dtype=tf.int32),
                    initial_state_,
                    initial_last_output,
                    initial_all_output, 
                    initial_gen_flag, 
                    initial_all_gen_flag, 
                    initial_all_cur_argmax, 
                    initial_last_cell_output))
       
        
        def fill_rest(i, all_output, all_gen_flag, all_cur_argmax):
                all_output = all_output.write(
                    i, tf.zeros((batch_size, self.feature_out_dim)))
                
                for j in range(self.sample_len):
                    all_gen_flag = all_gen_flag.write(
                        i * self.sample_len + j,
                        tf.zeros((batch_size, 1)))
                    all_cur_argmax = all_cur_argmax.write(
                        i * self.sample_len + j,
                        tf.zeros((batch_size,), dtype=tf.int64))
                
                return (i + 1,
                        all_output,
                        all_gen_flag,
                        all_cur_argmax)
        
        _, feature, gen_flag, cur_argmax = tf.while_loop(
                lambda a, b, c, d: a < time,
                fill_rest,
                (i, feature, gen_flag, cur_argmax))
        
        

        return feature, gen_flag, cur_argmax
        
        

class DoppelGANgerGeneratorMLP(tf.keras.layers.Layer):
    def __init__(self, addi_attribute_outputs, real_attribute_outputs,
                real_attribute_out_dim, addi_attribute_out_dim, real_attribute_mask,
                attribute_outputs,
                attribute=None, attribute_num_units=100, attribute_num_layers=3,
                *args, **kwargs):
        super(DoppelGANgerGeneratorMLP, self).__init__(*args, **kwargs)
        
        self.attribute = attribute
        self.__dict__['real_attribute_outputs'] = real_attribute_outputs
        self.__dict__['addi_attribute_outputs'] = addi_attribute_outputs
        
        self.real_attribute_out_dim = real_attribute_out_dim
        self.addi_attribute_out_dim = addi_attribute_out_dim
        self.real_attribute_mask = real_attribute_mask
        self.real_attribute_outputs = real_attribute_outputs
        self.attribute_outputs = attribute_outputs
        self.attribute_out_dim = np.sum([t.dim for t in attribute_outputs])
        if attribute is None:
            if len(self.addi_attribute_outputs) > 0:
                self.__dict__['all_attribute_outputs'] = \
                    [self.real_attribute_outputs,
                        self.addi_attribute_outputs]
            else:
                self.__dict__['all_attribute_outputs'] = [self.real_attribute_outputs]
        else:
            if len(self.addi_attribute_outputs) > 0:
                self.__dict__['all_attribute_outputs'] = \
                    [self.addi_attribute_outputs]
            else:
                self.__dict__['all_attribute_outputs'] = []
 
        self.attribute_num_units = attribute_num_units
        self.attribute_num_layers = attribute_num_layers
        self.layers_ = [[tf.keras.layers.Dense(self.attribute_num_units, activation='relu'), tf.keras.layers.BatchNormalization(momentum=0.9,
                                        epsilon=1e-5,
                                        scale=True,
                                        trainable=True)] for _ in range(self.attribute_num_layers -1)]
        
        self.initial_layers=[tf.keras.layers.Dense(self.attribute_num_units, activation='relu'), tf.keras.layers.Dense(self.attribute_num_units, activation='relu')]
        self.outputs={}
        for real_fake in range(len(self.__dict__['all_attribute_outputs'])):
            for output in range(len(self.__dict__['all_attribute_outputs'][real_fake])):
                self.outputs[str(real_fake) + ':' + str(output)] = tf.keras.layers.Dense(self.__dict__['all_attribute_outputs'][real_fake][output].dim)

        self.real_attribute_outputs = []
        self.addi_attribute_outputs = []
        self.real_attribute_out_dim = 0
        self.addi_attribute_out_dim = 0
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.real_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.real_attribute_out_dim += self.attribute_outputs[i].dim
            else:
                self.addi_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.addi_attribute_out_dim += \
                    self.attribute_outputs[i].dim

        for i in range(len(self.real_attribute_mask) - 1):
            if (self.real_attribute_mask[i] == False and
                    self.real_attribute_mask[i + 1] == True):
                raise Exception("Real attribute should come first")


        self.STR_REAL = "real"
        self.STR_ADDI = "addi"

    
    def call(self, attribute_input_noise, feature_input_noise, 
            addi_attribute_input_noise, training=None, attribute=None):
        batch_size = tf.shape(feature_input_noise)[0]
        
        #Prepare attribute variable names
        if attribute is None: #No respective attribute
            all_attribute = []
            all_discrete_attribute = []
            if len(self.addi_attribute_outputs) > 0:
                all_attribute_input_noise = \
                    [attribute_input_noise,
                        addi_attribute_input_noise]
                all_attribute_outputs = \
                    [self.real_attribute_outputs,
                        self.addi_attribute_outputs]
                all_attribute_part_name = \
                    [self.STR_REAL, self.STR_ADDI]
                all_attribute_out_dim = \
                    [self.real_attribute_out_dim,
                        self.addi_attribute_out_dim]
            else:
                all_attribute_input_noise = [attribute_input_noise]
                all_attribute_outputs = [self.real_attribute_outputs]
                all_attribute_part_name = [self.STR_REAL]
                all_attribute_out_dim = [self.real_attribute_out_dim]
            

        else: # generate with respect to attribute.
            all_attribute = [attribute]
            all_discrete_attribute = [attribute]
            if len(self.addi_attribute_outputs) > 0:
                all_attribute_input_noise = \
                    [addi_attribute_input_noise]
                all_attribute_outputs = \
                    [self.addi_attribute_outputs]
                all_attribute_part_name = \
                    [self.STR_ADDI]
                all_attribute_out_dim = [self.addi_attribute_out_dim]
            else:
                all_attribute_input_noise = []
                all_attribute_outputs = []
                all_attribute_part_name = []
                all_attribute_out_dim = []
        
        for part_i in range(len(all_attribute_input_noise)):
            
            #Only used if there's a respective attribute I think.
            if len(all_discrete_attribute) > 0:
                layers = [tf.concat(
                    [all_attribute_input_noise[part_i]] +
                    all_discrete_attribute,
                    axis=1)]
            #Layers is either additional or real    
            else:
                layers = [all_attribute_input_noise[part_i]]
            
            #build a feedforward network output FUNCTIONAL API
            for layer in self.layers_:
                try:
                    x = layer[0](x)#1st
                    x = layer[1](x, training = training)
                except Exception as e:
                    x = self.initial_layers[part_i](layers[-1])
                    #x = layer[0](x)# 2nd
                    x = layer[1](x)       
            
            part_attribute = []
            part_discrete_attribute = []
            #for each of the attribute outputs feed the noise through the MLP.
            for i in range(len(all_attribute_outputs[part_i])):
                output = all_attribute_outputs[part_i][i]              
               
                
                sub_output_ori = self.outputs[str(part_i) + ':' + str(i)](x) ##Not 100% sure this is doing the right thing
                
                if (output.type_ == OutputType.DISCRETE):
                    sub_output = tf.nn.softmax(sub_output_ori)
                    sub_output_discrete = tf.one_hot(
                        tf.argmax(sub_output, axis=1),
                        output.dim)
                elif (output.type_ == OutputType.CONTINUOUS):
                    if (output.normalization ==
                            Normalization.ZERO_ONE):
                        sub_output = tf.nn.sigmoid(
                            sub_output_ori)
                    elif (output.normalization ==
                            Normalization.MINUSONE_ONE):
                        sub_output = tf.nn.tanh(sub_output_ori)
                    else:
                        raise Exception("unknown normalization"
                                        " type")
                    sub_output_discrete = sub_output
                else:
                    raise Exception("unknown output type")
                
                part_attribute.append(sub_output)
                part_discrete_attribute.append(
                    sub_output_discrete)
            

            part_attribute = tf.concat(part_attribute, axis=1)
            part_discrete_attribute = tf.concat(
                part_discrete_attribute, axis=1)
            part_attribute = tf.reshape(
                part_attribute,
                [-1, all_attribute_out_dim[part_i]])

            
            part_discrete_attribute = tf.reshape(
                part_discrete_attribute,
                [-1, all_attribute_out_dim[part_i]])
            
            # batch_size * dim
        
            part_discrete_attribute = tf.dtypes.cast(tf.stop_gradient(
                part_discrete_attribute), tf.float64)

            all_attribute.append(part_attribute)
            all_discrete_attribute.append(part_discrete_attribute)
            del x

        all_attribute = tf.concat(all_attribute, axis=1)
    
        all_discrete_attribute = tf.concat(all_discrete_attribute, axis=1)
        all_attribute = tf.reshape(
            all_attribute,
            [-1, self.attribute_out_dim])
        all_discrete_attribute = tf.reshape(
            all_discrete_attribute,
            [-1, self.attribute_out_dim])

        return all_attribute, all_discrete_attribute  

class DoppelGANgerGenerator(tf.keras.Model):
    def __init__(self, feed_back, noise,
                 feature_outputs, attribute_outputs, real_attribute_mask,
                 sample_len,
                 attribute_num_units=100, attribute_num_layers=3,
                 feature_num_units=100, feature_num_layers=1,
                 initial_state=RNNInitialStateType.RANDOM,
                 initial_stddev=0.02, *args, **kwargs):
        super(DoppelGANgerGenerator, self).__init__(*args, **kwargs)
        self.feed_back = feed_back
        self.noise = noise
        self.attribute_num_units = attribute_num_units
        self.attribute_num_layers = attribute_num_layers
        self.feature_num_units = feature_num_units
        self.feature_outputs = feature_outputs
        self.attribute_outputs = attribute_outputs
        self.real_attribute_mask = real_attribute_mask
        self.feature_num_layers = feature_num_layers
        self.sample_len = sample_len
        self.initial_state = initial_state
        self.initial_stddev = initial_stddev
        self.feature_out_dim = (np.sum([t.dim for t in feature_outputs]) *
                                self.sample_len)
        self.attribute_out_dim = np.sum([t.dim for t in attribute_outputs])
        if not self.noise and not self.feed_back:
            raise Exception("noise and feed_back should have at least "
                            "one True")
        
        self.real_attribute_outputs = []
        self.addi_attribute_outputs = []
        self.real_attribute_out_dim = 0
        self.addi_attribute_out_dim = 0

        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.real_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.real_attribute_out_dim += self.attribute_outputs[i].dim
            else:
                self.addi_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.addi_attribute_out_dim += \
                    self.attribute_outputs[i].dim
        for i in range(len(self.real_attribute_mask) - 1):
            if (self.real_attribute_mask[i] == False and
                    self.real_attribute_mask[i + 1] == True):
                raise Exception("Real attribute should come first")
                
        self.gen_flag_id = None
        for i in range(len(self.feature_outputs)):
            if self.feature_outputs[i].is_gen_flag:
                self.gen_flag_id = i
                break
        if self.gen_flag_id is None:
            raise Exception("cannot find gen_flag_id")
        if self.feature_outputs[self.gen_flag_id].dim != 2:
            raise Exception("gen flag output's dim should be 2")

        
        

        self.rnn = DoppelGANgerGeneratorRNN(self.feature_outputs, self.sample_len, True)
        self.MLP = DoppelGANgerGeneratorMLP(self.addi_attribute_outputs, self.real_attribute_outputs,
                                            self.real_attribute_out_dim, self.addi_attribute_out_dim,
                                            self.real_attribute_mask, self.attribute_outputs)

    def call(self, attribute_input_noise, addi_attribute_input_noise,    #def call()? 
            feature_input_noise, feature_input_data, train,
            attribute=None):
        
                
        all_attribute, all_discrete_attribute = self.MLP(attribute_input_noise, feature_input_noise, 
                                                        addi_attribute_input_noise)
        
        
        feature, gen_flag, cur_argmax = self.rnn(all_discrete_attribute, feature_input_noise, 
                                            feature_input_data)
        
        time = tf.convert_to_tensor(feature_input_noise).get_shape().as_list()[1]
        
        if time is None:
            time = tf.shape(feature_input_noise)[1]        
        batch_size = tf.shape(feature_input_noise)[0]

        feature = feature.stack()
            
        # time * batch_size * (dim * sample_len)
        gen_flag = gen_flag.stack()
        # (time * sample_len) * batch_size * 1
        cur_argmax = cur_argmax.stack()

        gen_flag = tf.transpose(gen_flag, [1, 0, 2])
        # batch_size * (time * sample_len) * 1
        cur_argmax = tf.transpose(cur_argmax, [1, 0])
        # batch_size * (time * sample_len)
        length = tf.reduce_sum(gen_flag, [1, 2])
        # batch_size

        feature = tf.transpose(feature, [1, 0, 2])
        # batch_size * time * (dim * sample_len)
        gen_flag_t = tf.reshape(
            gen_flag,
            [batch_size, time, self.sample_len])
        # batch_size * time * sample_len
        gen_flag_t = tf.reduce_sum(gen_flag_t, [2])
        # batch_size * time
        gen_flag_t = tf.cast((gen_flag_t > 0.5), dtype=tf.float32)
        gen_flag_t = tf.expand_dims(gen_flag_t, 2)
        # batch_size * time * 1
        gen_flag_t = tf.tile(
            gen_flag_t,
            [1, 1, self.feature_out_dim])
        # batch_size * time * (dim * sample_len)
        # zero out the parts after sequence ends
        feature = feature * gen_flag_t
        feature = tf.reshape(
            feature,
            [batch_size,
                time * self.sample_len,
                self.feature_out_dim / self.sample_len])
        # batch_size * (time * sample_len) * dim
    
        
        
        #sys.exit("FINSIHED")
        return feature, all_attribute, gen_flag, length, cur_argmax


    
# from load_data import *
# from util import *
# seq_len = 7
# (data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = load_data("data")

# print("-----DATA LOADING PART-----")
# print(data_feature.shape)
# print(data_attribute.shape)
# print(data_gen_flag.shape)
# num_real_attribute = len(data_attribute_outputs)

# (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = \
#     normalize_per_sample(data_feature, data_attribute, data_feature_outputs,data_attribute_outputs)

# print("-----DATA NORMALIZATION PART-----")
# print(real_attribute_mask)
# print(data_feature.shape)
# print(data_attribute.shape)
# print(len(data_attribute_outputs))

# print("-----ADD GEN FLAG PART -----")
# data_feature, data_feature_outputs = add_gen_flag(
#         data_feature, data_gen_flag, data_feature_outputs, seq_len)
# print(data_feature.shape)
# print(len(data_feature_outputs))

# generator = DoppelGANgerGenerator(
#         feed_back=False,
#         noise=True,
#         feature_outputs=data_feature_outputs,
#         attribute_outputs=data_attribute_outputs,
#         real_attribute_mask=real_attribute_mask,
#         sample_len=seq_len)

# print("done")

# # g_real_attribute_input_noise_train_pl_l = np.ones((10, 5)) *0.9
# # g_addi_attribute_input_noise_train_pl_l = np.ones((10, 5)) *0.9
# # g_feature_input_noise_train_pl_l = np.ones((10, 1, 5)) *0.9
# # g_feature_input_data_train_pl_l = np.ones((10, 28)) *0.9

# g_feature_input_noise_train_pl_l = np.random.normal(size=[10, 1, 5]).astype(np.float32)
# g_real_attribute_input_noise_train_pl_l = np.random.normal(size=[10, 5]).astype(np.float32)
# g_addi_attribute_input_noise_train_pl_l = np.random.normal(size=[10, 5]).astype(np.float32)
# g_feature_input_data_train_pl_l = np.zeros([10, 28 ]).astype(np.float32)


# (g_output_feature_train_tf, g_output_attribute_train_tf,
#              g_output_gen_flag_train_tf, g_output_length_train_tf,
#              g_output_argmax_train_tf) = \
#                 generator(
#                     g_real_attribute_input_noise_train_pl_l,
#                     g_addi_attribute_input_noise_train_pl_l,
#                     g_feature_input_noise_train_pl_l,
#                     g_feature_input_data_train_pl_l,
#                     train=True)

# print("g out")
# print(g_output_feature_train_tf)

# print("done")