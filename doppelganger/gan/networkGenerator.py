import tensorflow as tf
from op import linear, batch_norm, flatten
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
from data_loading import sine_data_generation_f_a, real_data_loading_prism, renormalize
tf.keras.backend.set_floatx('float64')

class RNNInitialStateType(Enum):
    ZERO = "ZERO"
    RANDOM = "RANDOM"
    VARIABLE = "VARIABLE"


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

        self.STR_REAL = "real"
        self.STR_ADDI = "addi"

    def build(self, attribute_input_noise, addi_attribute_input_noise,    #def call()? 
            feature_input_noise, feature_input_data, train,
            attribute=None):
        
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
        
        #for both the additional and real attributes:
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
            
            inputs = tf.keras.Input(shape=layers[-1].shape[-1]) # added [-1] as a hacky fix, the shape wasn't working properly ?? 
            x = tf.keras.layers.Dense(self.attribute_num_units, activation='relu')(inputs)
            x = tf.keras.layers.BatchNormalization(momentum=0.9,
                                            epsilon=1e-5,
                                            scale=True,
                                            trainable=True)(x)
            for _ in range(self.attribute_num_layers-2):
                x = tf.keras.layers.Dense(self.attribute_num_units, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization(momentum=0.9,
                                            epsilon=1e-5,
                                            scale=True,
                                            trainable=True)(x)

                
            
             #SEQUENTIAL API
            # model = tf.keras.Sequential()
            # for i in range(self.attribute_num_layers - 1):
            #     model.add(tf.keras.layers.Dense(self.attribute_num_units)),
            #     model.add(tf.keras.layers.ReLU()),
            #     model.add(tf.keras.layers.BatchNormalization(momentum=0.9,
            #                                 epsilon=1e-5,
            #                                 scale=True,
            #                                 trainable=True))
            

           
            part_attribute = []
            part_discrete_attribute = []
            #for each of the attribute outputs feed the noise through the MLP.
            for i in range(len(all_attribute_outputs[part_i])):
                output = all_attribute_outputs[part_i][i]
                
                outputs = tf.keras.layers.Dense(output.dim)(x)
                model = tf.keras.Model(inputs, outputs, name="nn")
               
                
                sub_output_ori = model(layers[-1]) ##Not 100% sure this is doing the right thing
                
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
                [batch_size, all_attribute_out_dim[part_i]])

            
            part_discrete_attribute = tf.reshape(
                part_discrete_attribute,
                [batch_size, all_attribute_out_dim[part_i]])
            
            # batch_size * dim
        
            part_discrete_attribute = tf.dtypes.cast(tf.stop_gradient(
                part_discrete_attribute), tf.float64)

            all_attribute.append(part_attribute)
            all_discrete_attribute.append(part_discrete_attribute)


        all_attribute = tf.concat(all_attribute, axis=1)
    
        all_discrete_attribute = tf.concat(all_discrete_attribute, axis=1)
        all_attribute = tf.reshape(
            all_attribute,
            [batch_size, self.attribute_out_dim])
        all_discrete_attribute = tf.reshape(
            all_discrete_attribute,
            [batch_size, self.attribute_out_dim])
        
        # rnn_network = tf.keras.Sequential([
        
        # tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(self.feature_num_units) for _ in range(self.feature_num_layers)]))
        
        # ])
        
        rnn_network =  tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(self.feature_num_units) for _ in range(self.feature_num_layers)]), return_state=True)

        #rnn_network = tf.compat.v1.nn.rnn_cell.MultiRNNCell([tf.keras.layers.LSTMCell(self.feature_num_units) for _ in range(self.feature_num_layers)])

        feature_input_data_dim = \
                len(tf.convert_to_tensor(feature_input_data).get_shape().as_list())
        if feature_input_data_dim == 3:
            feature_input_data_reshape = tf.transpose(
                feature_input_data, [1, 0, 2])
        feature_input_noise_reshape = tf.transpose(
            feature_input_noise, [1, 0, 2])


        if self.initial_state == RNNInitialStateType.ZERO:
            initial_state = rnn_network.zero_state(
                batch_size, tf.float64)
        elif self.initial_state == RNNInitialStateType.RANDOM:
            
            input_ = tf.random.normal([1, 100, 1])
            
            _, initial_state = rnn_network(input_)
            
            # initial_state = tf.random.normal(
            #     shape=(self.feature_num_layers,
            #             2,
            #             batch_size,
            #             self.feature_num_units),
            #     mean=0.0, stddev=1.0)
            
            # initial_state = tf.unstack(initial_state, axis=0)
            
            # initial_state = tuple(
            #     [tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
            #         initial_state[idx][0], initial_state[idx][1])
            #         for idx in range(self.feature_num_layers)])
        elif self.initial_state == RNNInitialStateType.VARIABLE: ### this isn't upgraded
            initial_state = []
            for i in range(self.feature_num_layers):
                sub_initial_state1 = tf.get_variable(
                    "layer{}_initial_state1".format(i),
                    (1, self.feature_num_units),
                    initializer=tf.random_normal_initializer(
                        stddev=self.initial_stddev))
                sub_initial_state1 = tf.tile(
                    sub_initial_state1, (batch_size, 1))
                sub_initial_state2 = tf.get_variable(
                    "layer{}_initial_state2".format(i),
                    (1, self.feature_num_units),
                    initializer=tf.random_normal_initializer(
                        stddev=self.initial_stddev))
                sub_initial_state2 = tf.tile(
                    sub_initial_state2, (batch_size, 1))
                sub_initial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                    sub_initial_state1, sub_initial_state2)
                initial_state.append(sub_initial_state)
            initial_state = tuple(initial_state)
        else:
            raise NotImplementedError
        
        time = tf.convert_to_tensor(feature_input_noise).get_shape().as_list()[1]
        
        if time is None:
            time = tf.shape(feature_input_noise)[1]
        #@tf.function()

        #Feeds in the attribute generations into an RNN as well as the previous state to build up a time series
        def compute(time):
            i=0
            state, new_state = initial_state, initial_state
            all_output = tf.TensorArray(tf.float32, time)
            gen_flag = tf.ones((batch_size, 1))
            all_gen_flag = tf.TensorArray(tf.float32, time * self.sample_len)
            all_cur_argmax = tf.TensorArray(tf.int64, time * self.sample_len)
            

            if feature_input_data_dim == 2:
                last_output =  feature_input_data
            else:
                last_output = feature_input_data_reshape[0]
            
            while i < time and tf.equal(tf.reduce_max(gen_flag), 1):

            
                input_all = [all_discrete_attribute]
                
                
                
                if self.noise:
                    input_all.append(feature_input_noise_reshape[i])
                    
                if self.feed_back:
                    if feature_input_data_dim == 3:
                        input_all.append(feature_input_data_reshape[i])
                    else:
                        input_all.append(last_output)
                input_all = tf.concat(input_all, axis=1)
                
                input_all = tf.expand_dims(input_all, axis=2)
                
                
                
                rnn_network =  tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(self.feature_num_units) for _ in range(self.feature_num_layers)]), return_state=True)          
                
                
                
                cell_new_output, new_state = rnn_network(input_all)#, initial_state=state)
                
                
                new_output_all = []
                id_ = 0

                for j in range(self.sample_len):
                    for k in range(len(self.feature_outputs)):
                        output = self.feature_outputs[k]
                        sub_output = tf.keras.layers.Dense(output.dim)(cell_new_output)
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
                    
                
                state = new_state
                last_output = new_output
                all_output = all_output.write(i, tf.cast(new_output, dtype=tf.float32))
                last_cell_output = cell_new_output
                i+=1
            return (i + 1,
                        new_state,
                        new_output,
                        all_output.write(i-1, tf.cast(new_output, dtype=tf.float32)),
                        gen_flag,
                        all_gen_flag,
                        all_cur_argmax,
                        cell_new_output)
        
        (i, state, _, feature, _, gen_flag, cur_argmax,
                cell_output) = compute(time)
        
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



### TESTING ###


sample_len = 130
auto_normalize=True

data_feature_outputs = [
    output.Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
    output.Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
    output.Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
    output.Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False),
    output.Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.ZERO_ONE, is_gen_flag=False)
]

data_attribute_outputs = [
    Output(type_=OutputType.DISCRETE, dim=330, is_gen_flag=False)

]
num_real_attribute = len(data_attribute_outputs)
data_feature, data_attribute, data_gen_flag, min_, max_ = real_data_loading_prism()
no, seq_len, dim = data_feature.shape[0], data_feature.shape[1], data_feature.shape[2]






if auto_normalize:
    (data_feature, data_attribute, data_attribute_outputs,
    real_attribute_mask) = \
        normalize_per_sample(
            data_feature, data_attribute, data_feature_outputs,
            data_attribute_outputs)
else:
    real_attribute_mask = [True] * len(data_attribute_outputs)





data_feature, data_feature_outputs = add_gen_flag(
    data_feature, data_gen_flag, data_feature_outputs, sample_len)


generator = DoppelGANgerGenerator(
        feed_back=False,
        noise=True,
        feature_outputs=data_feature_outputs,
        attribute_outputs=data_attribute_outputs,
        real_attribute_mask=real_attribute_mask,
        sample_len=sample_len)


g_real_attribute_input_noise_train_pl_l = np.ones((1, 5))
g_addi_attribute_input_noise_train_pl_l = np.ones((1, 5)) 
g_feature_input_noise_train_pl_l = np.ones((1, 1, 5))
g_feature_input_data_train_pl_l = np.ones((1, 910))

(g_output_feature_train_tf, g_output_attribute_train_tf,
             g_output_gen_flag_train_tf, g_output_length_train_tf,
             g_output_argmax_train_tf) = \
                generator.build(
                    g_real_attribute_input_noise_train_pl_l,
                    g_addi_attribute_input_noise_train_pl_l,
                    g_feature_input_noise_train_pl_l,
                    g_feature_input_data_train_pl_l,
                    train=True)

print("FINISHED")
### END TESTING ###