import tensorflow as tf
from tensorflow.keras import layers
from output import OutputType, Normalization, Output
from util import *
from load_data import *
import numpy as np
from enum import Enum
import os

num_layers = 5
seq_len = 7
ori_features_dim = 2
features_dim = 4 #after some magic by utils, dim becomes 4??
ori_attributes_dim = 2
attributes_dim = 6 #after some magic by utils, dim becomes 6??
total_generate_num_sample = 500

#discriminator
input_feature_d = tf.keras.Input(shape=(seq_len, features_dim)) #TODO: get proper features_dim
input_attribute_d = tf.keras.Input(shape=(attributes_dim)) #TODO: get proper attributes_dim
input_feature_fl_d = tf.keras.layers.Flatten()(input_feature_d)
input_attribute_fl_d = tf.keras.layers.Flatten()(input_attribute_d)
output_d = tf.keras.layers.Concatenate(axis=1)([input_feature_fl_d, input_attribute_fl_d])
for i in range(num_layers - 1):
    output_ = tf.keras.layers.Dense(200, activation='relu')(output_d)
output_d = tf.keras.layers.Dense(1)(output_d)
output_d = tf.squeeze(output_d, axis=1)
discriminator_model = tf.keras.models.Model(inputs=[input_feature_d, input_attribute_d], outputs=output_d)
#discriminator_model.summary()

#AttrDiscriminator
input_attribute_ad = tf.keras.Input(shape=(attributes_dim))
output_ad = tf.keras.layers.Flatten()(input_attribute_ad)
for i in range(num_layers - 1):
    output_ad = tf.keras.layers.Dense(200, activation='relu')(output_ad)
output_ad = tf.keras.layers.Dense(1)(output_ad)
output_ad = tf.squeeze(output_ad, axis=1)
attrdiscriminator_model = tf.keras.models.Model(inputs=input_attribute_ad, outputs=output_ad)
#attrdiscriminator_model.summary()

# class DoppelGANgerGenerator(tf.keras.Model):
#     def __init__(self, feed_back, noise,
#                  feature_outputs, attribute_outputs, real_attribute_mask,
#                  seq_len,
#                  attribute_num_units=100, attribute_num_layers=3,
#                  feature_num_units=100, feature_num_layers=1,
#                  initial_stddev=0.02, *args, **kwargs) #some other stuff
#         super(DoppelGANgerGenerator, self).__init__(*args, **kwargs)
#         self.feed_back = feed_back
#         self.noise = noise
#         self.attribute_num_units = attribute_num_units
#         self.attribute_num_layers = attribute_num_layers
#         self.feature_num_units = feature_num_units
#         self.feature_outputs = feature_outputs
#         self.attribute_outputs = attribute_outputs
#         self.real_attribute_mask = real_attribute_mask
#         self.feature_num_layers = feature_num_layers
#         self.sample_len = sample_len
#         self.initial_state = initial_state
#         self.initial_stddev = initial_stddev
#         self.feature_out_dim = (np.sum([t.dim for t in feature_outputs]) * self.seq_len)
#         self.attribute_out_dim = np.sum([t.dim for t in attribute_outputs])
#         if not self.noise and not self.feed_back:
#             raise Exception("noise and feed_back should have at least "
#                             "one True")

#         self.real_attribute_outputs = []
#         self.addi_attribute_outputs = []
#         self.real_attribute_out_dim = 0
#         self.addi_attribute_out_dim = 0

#         for i in range(len(self.real_attribute_mask)):
#             if self.real_attribute_mask[i]:
#                 self.real_attribute_outputs.append(self.attribute_outputs[i])
#                 self.real_attribute_out_dim += self.attribute_outputs[i].dim
#             else:
#                 self.addi_attribute_outputs.append(self.attribute_outputs[i])
#                 self.addi_attribute_out_dim += self.attribute_outputs[i].dim
        
#         for i in range(len(self.real_attribute_mask) - 1):
#             if (self.real_attribute_mask[i] == False and self.real_attribute_mask[i + 1] == True):
#                 raise Exception("Real attribute should come first")

#         self.gen_flag_id = None
#         for i in range(len(self.feature_outputs)):
#             if self.feature_outputs[i].is_gen_flag:
#                 self.gen_flag_id = i
#                 break
#         if self.gen_flag_id is None:
#             raise Exception("cannot find gen_flag_id")
#         if self.feature_outputs[self.gen_flag_id].dim != 2:
#             raise Exception("gen flag output's dim should be 2")

#         self.STR_REAL = "real"
#         self.STR_ADDI = "addi"

#         self.concat = tf.keras.layers.Concatenate(axis=1)
#         self.dense = tf.keras.layers.Dense(self.attribute_num_layers, activation='relu')
#         self.batchnorm = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

#     def build(self, attribute_input_noise, addi_attribute_input_noise,
#               feature_input_noise, feature_input_data, train,
#               attribute=None):

#         batch_size = tf.shape(feature_input_noise)[0]
        
#         if attribute is None:
#             all_attribute = []
#             all_discrete_attribute = []
#             if len(self.addi_attribute_outputs) > 0:
#                 all_attribute_input_noise = [attribute_input_noise, addi_attribute_input_noise]
#                 all_attribute_outputs = [self.real_attribute_outputs, self.addi_attribute_outputs]
#                 all_attribute_part_name = [self.STR_REAL, self.STR_ADDI]
#                 all_attribute_out_dim = [self.real_attribute_out_dim, self.addi_attribute_out_dim]
#             else:
#                 all_attribute_input_noise = [attribute_input_noise]
#                 all_attribute_outputs = [self.real_attribute_outputs]
#                 all_attribute_part_name = [self.STR_REAL]
#                 all_attribute_out_dim = [self.real_attribute_out_dim]
#         else:
#             all_attribute = [attribute]
#             all_discrete_attribute = [attribute]
#             if len(self.addi_attribute_outputs) > 0:
#                 all_attribute_input_noise = [addi_attribute_input_noise]
#                 all_attribute_outputs = [self.addi_attribute_outputs]
#                 all_attribute_part_name = [self.STR_ADDI]
#                 all_attribute_out_dim = [self.addi_attribute_out_dim]
#             else:
#                 all_attribute_input_noise = []
#                 all_attribute_outputs = []
#                 all_attribute_part_name = []
#                 all_attribute_out_dim = []
        
#         for part_i in range(len(all_attribute_input_noise)):
#             if len(all_discrete_attribute) > 0:
#                 input_g = self.concat([all_attribute_input_noise[part_i], all_discrete_attribute])
#             else:
#                 input_g = all_attribute_input_noise[part_i]
#             for i in range(self.attribute_num_layers - 1):
#                 input_g = self.dense(input_g)
#             output_g = self.batchnorm(input_g)

#             part_attribute = []
#             part_discrete_attribute = []

#             for i in range(len(all_attribute_outputs[part_i])):
#                 output = all_attribute_outputs[part_i][i]

#                 sub_output_ori = tf.keras.layers.Dense(output.dim)(output)

#                 if output.type_ == OutputType.DISCRETE:
#                     sub_output = tf.keras.layers.Softmax(sub_output_ori)

(data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = load_data("data")

print(data_feature.shape)
print(data_attribute.shape)
print(data_gen_flag.shape)

(data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = \
    normalize_per_sample(data_feature, data_attribute, data_feature_outputs,data_attribute_outputs)

print(real_attribute_mask)
print(data_feature.shape)
print(data_attribute.shape)
print(len(data_attribute_outputs))

data_feature, data_feature_outputs = add_gen_flag(
        data_feature, data_gen_flag, data_feature_outputs, seq_len)
print(data_feature.shape)
print(len(data_feature_outputs))

g_attribute_num_units = 100
g_attribute_num_layers = 3
g_feature_num_units = 100
g_feature_num_layers = 1
g_initital_stddev = 0.02
feed_back = False
noise = True
feature_outputs = data_feature_outputs
attribute_outputs = data_attribute_outputs
attribute = None
feature_latent_dim = 5
attribute_latent_dim = 5

#defined in a fn in doppelganger
length_ = int(data_feature.shape[1] / seq_len)
sample_feature_dim = data_feature.shape[2]
feature_input_noise = np.random.normal(size=[total_generate_num_sample, length_, feature_latent_dim]).astype(np.float32)
attribute_input_noise = np.random.normal(size=[total_generate_num_sample, attribute_latent_dim])
addi_attribute_input_noise = np.random.normal(size=[total_generate_num_sample, attribute_latent_dim])
feature_input_data = np.zeros([total_generate_num_sample, seq_len *sample_feature_dim ])

#real_attribute_mask = None
feature_out_dim = (np.sum([t.dim for t in feature_outputs]) * seq_len)
attribute_out_dim = np.sum([t.dim for t in attribute_outputs])
STR_REAL = "real"
STR_ADDI = "addi"

if not noise and not feed_back:
    raise Exception("noise and feed_back should have at least ""one True")

#SOME OTHER INIT STUFF THAT I'M SKIPPING FOR NOW

real_attribute_outputs = []
addi_attribute_outputs = []
real_attribute_out_dim = 0
addi_attribute_out_dim = 0

for i in range(len(real_attribute_mask)):
    if real_attribute_mask[i]:
        real_attribute_outputs.append(attribute_outputs[i])
        real_attribute_out_dim += attribute_outputs[i].dim
    else:
        addi_attribute_outputs.append(attribute_outputs[i])
        addi_attribute_out_dim += attribute_outputs[i].dim

for i in range(len(real_attribute_mask) - 1):
    if (real_attribute_mask[i] == False and real_attribute_mask[i + 1] == True):
        raise Exception("Real attribute should come first")

gen_flag_id = None
for i in range(len(feature_outputs)):
    if feature_outputs[i].is_gen_flag:
        gen_flag_id = i
        break
if gen_flag_id is None:
    raise Exception("cannot find gen_flag_id")
if feature_outputs[gen_flag_id].dim != 2:
    raise Exception("gen flag output's dim should be 2")


batch_size = tf.shape(feature_input_noise)[0]

####BUILD
if attribute is None:
    all_attribute = []
    all_discrete_attribute = []
    if len(addi_attribute_outputs) > 0:
        all_attribute_input_noise = [attribute_input_noise, addi_attribute_input_noise]
        all_attribute_outputs = [real_attribute_outputs, addi_attribute_outputs]
        all_attribute_part_name = [STR_REAL, STR_ADDI]
        all_attribute_out_dim = [real_attribute_out_dim, addi_attribute_out_dim]
    else:
        all_attribute_input_noise = [attribute_input_noise]
        all_attribute_outputs = [real_attribute_outputs]
        all_attribute_part_name = [STR_REAL]
        all_attribute_out_dim = [real_attribute_out_dim]
else:
    all_attribute = [attribute]
    all_discrete_attribute = [attribute]
    if len(addi_attribute_outputs) > 0:
        all_attribute_input_noise = [addi_attribute_input_noise]
        all_attribute_outputs = [addi_attribute_outputs]
        all_attribute_part_name = [STR_ADDI]
        all_attribute_out_dim = [addi_attribute_out_dim]
    else:
        all_attribute_input_noise = []
        all_attribute_outputs = []
        all_attribute_part_name = []
        all_attribute_out_dim = []

for part_i in range(len(all_attribute_input_noise)):
    if len(all_discrete_attribute) > 0:
        input_g = tf.keras.layers.Concatenate(axis=1)([all_attribute_input_noise[part_i], np.squeeze(np.asarray(all_discrete_attribute))])
    else:
        input_g = all_attribute_input_noise[part_i]
    for i in range(g_attribute_num_layers - 1):
        input_g = tf.keras.layers.Dense(g_attribute_num_units)(input_g)
    output_g = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(input_g)

    part_attribute = []
    part_discrete_attribute = []

    for i in range(len(all_attribute_outputs[part_i])):
        output = all_attribute_outputs[part_i][i]

        sub_output_ori = tf.keras.layers.Dense(output.dim)(output_g) #accepting output or output_g?

        if output.type_ == OutputType.DISCRETE:
            sub_output = tf.keras.layers.Softmax()(sub_output_ori) #layers or activation?
            sub_output_discrete = tf.one_hot(tf.math.argmax(sub_output, axis=1), output.dim)
        elif output.type_ == OutputType.CONTINUOUS:
            if output.normalization == Normalization.ZERO_ONE:
                sub_output = tf.keras.activations.sigmoid(sub_output_ori)
            elif output.normalization == Normalization.MINUSONE_ONE:
                sub_output = tf.keras.activations.tanh(sub_output_ori)
            else:
                raise Exception("unknown normalization type")
            sub_output_discrete = sub_output
        else:
            raise Exception("unknown output type")
        part_attribute.append(sub_output)
        part_discrete_attribute.append(sub_output_discrete)

    # check indentation?
    part_attribute = tf.concat(part_attribute, axis=1)
    part_discrete_attribute = tf.concat(part_discrete_attribute, axis=1)
    part_attribute = tf.reshape(part_attribute,[batch_size, all_attribute_out_dim[part_i]])
    part_discrete_attribute = tf.reshape(part_discrete_attribute, [batch_size, all_attribute_out_dim[part_i]])

    part_discrete_attribute = tf.stop_gradient(part_discrete_attribute)

    all_attribute.append(part_attribute)
    all_discrete_attribute.append(part_discrete_attribute)

all_attribute = tf.concat(all_attribute, axis=1)
all_discrete_attribute = tf.concat(all_discrete_attribute, axis=1)
all_attribute = tf.reshape(all_attribute,[batch_size, attribute_out_dim])
all_discrete_attribute = tf.reshape(all_discrete_attribute,[batch_size, attribute_out_dim])

#TODO: 
# not even sure if this is right? 
# this is later called with input_all (shape: (?,11)), state (shape: (?,100), (?,100))
cells =  [tf.keras.layers.LSTMCell(g_feature_num_units) for i in range(g_feature_num_layers)]
rnn_network = tf.keras.layers.StackedRNNCells(cells)
initial_state = rnn_network.get_initial_state(batch_size=batch_size, dtype=tf.float32)

# rnn_network = tf.keras.Sequential(name = 'rnn_network')
# for i in range(g_feature_num_layers):
#     rnn_network.add(tf.keras.layers.LSTM(g_feature_num_units, return_sequences=True, return_state= True))
# rnn_network.summary()

feature_input_data_dim = len(feature_input_data.shape)
if feature_input_data_dim == 3:
    feature_input_data_reshape = tf.transpose(feature_input_data, [1, 0, 2])
feature_input_noise_reshape = tf.transpose(feature_input_noise, [1, 0, 2])  

time = feature_input_noise.shape[1]
if time is None:
    time = tf.shape(feature_input_noise)[1]

### COMPUTE
## this has not been tested
def compute(i, state, last_output, all_output,gen_flag, all_gen_flag, all_cur_argmax, last_cell_output):
    input_all = [all_discrete_attribute]
    if noise:
        input_all.append(feature_input_noise_reshape[i])
    if feed_back:
        if feature_input_data_dim == 3:
            input_all.append(feature_input_data_reshape[i])
        else:
            input_all.append(last_output)
    input_all = tf.concat(input_all, axis=1)

    cell_new_output, new_state = rnn_network(input_all, state) #state???
    new_output_all = []
    id_ = 0
    for j in range(seq_len):
        for k in range(len(feature_outputs)): #feature_outputs is not passed into the fn
            output = feature_outputs[k]

            sub_output = tf.keras.layers.Dense(output.dim)(cell_new_output)
            if output.type_ == OutputType.DISCRETE:
                sub_output = tf.keras.layers.Softmax()(sub_output)
            elif output.type_ == OutputType.CONTINUOUS:
                if output.normalization == Normalization.ZERO_ONE:
                    sub_output = tf.keras.activations.sigmoid(sub_output)
                elif output.normalization == Normalization.MINUSONE_ONE:
                    sub_output = tf.keras.activations.tanh(sub_output_ori)
                else:
                    raise Exception("unknown normalization type")
            else:
                raise Exception("unknown output type")
            new_output_all.append(sub_output)
            id_ += 1
    new_output = tf.concat(new_output_all, axis=1) #shape(?,28)

    for j in range(seq_len):
        all_gen_flag = all_gen_flag.write( i * seq_len + j, gen_flag)
        cur_gen_flag = tf.cast(tf.math.equal(tf.math.argmax(new_output_all[(j*len(feature_outputs)+gen_flag_id)], axis=1), 0), tf.float32)
        cur_gen_flag = tf.reshape(cur_gen_flag, [-1, 1])
        all_cur_argmax = all_cur_argmax.write(i*seq_len+j, tf.argmax(new_output_all[(j*len(feature_outputs)+gen_flag_id)], axis=1))
        gen_flag = gen_flag * cur_gen_flag #shape(?,1)

    return (i+1, new_state, new_output, all_output.write(i, new_output),
            gen_flag, all_gen_flag, all_cur_argmax, cell_new_output)
    #new_state- LSTM State Tuple
    #new_output- shape(?,28)
    #cell_new_output- shape(?,100)

#initial_state = tf.zeros(batch_size, tf.float32) #??

#TODO: sth wrong with RNN which idk how to fix so this does not work
# which makes everything else below not work since feature... is not defined
(i, state, _, feature, _, gen_flag, cur_argmax, cell_output) = \
    tf.while_loop(lambda a, b, c, d, e, f, g, h:
    tf.math.logical_and(a < time, tf.equal(tf.reduce_max(e), 1)),
    compute, 
    (0, 
    initial_state, 
    feature_input_data if feature_input_data_dim==2 else feature_input_data_reshape[0],
    tf.TensorArray(tf.float32, time),
    tf.ones((batch_size, 1)),
    tf.TensorArray(tf.float32, time*seq_len),
    tf.TensorArray(tf.int64, time*seq_len),
    tf.zeros((batch_size, g_feature_num_units))))

def fill_rest(i, all_output, all_gen_flag, all_cur_argmax):
    all_output = all_output.write(i, tf.zeros((batch_size, feature_out_dim)))
    for j in range(seq_len):
        all_gen_flag = all_gen_flag.write(i * seq_len + j, tf.zeros((batch_size, 1)))
        all_cur_argmax = all_cur_argmax.write(i * seq_len + j, tf.zeros((batch_size,), dtype=tf.int64))
    return (i+1, all_output, all_gen_flag, all_cur_argmax)

_, feature, gen_flag, cur_argmax = tf.while_loop(lambda a, b, c, d: 
                                    a < time, fill_rest, (i, feature, gen_flag, cur_argmax))

feature = feature.stack()
# time * batch_size * (dim * sample_len)
gen_flag = gen_flag.stack()
# (time * sample_len) * batch_size * 1
cur_argmax = cur_argmax.stack()

gen_flag = tf.transpose(gen_flag, [1, 0, 2])
# batch_size * (time * sample_len) * 1
cur_argmax = tf.transpose(cur_argmax, [1, 0])
# batch_size * (time * sample_len)
length = tf.math.reduce_sum(gen_flag, [1, 2])
# batch_size

feature = tf.transpose(feature, [1, 0, 2])
# batch_size * time * (dim * sample_len)
gen_flag_t = tf.reshape(gen_flag, [batch_size, time, seq_len])
# batch_size * time * sample_len
gen_flag_t = tf.reduce_sum(gen_flag_t, [2])
# batch_size * time
gen_flag_t = tf.cast(gen_flag_t > 0.5, tf.float32)
gen_flag_t = tf.expand_dims(gen_flag_t, 2)
# batch_size * time * 1
gen_flag_t = tf.tile(gen_flag_t,[1, 1, feature_out_dim])
# batch_size * time * (dim * sample_len)
# zero out the parts after sequence ends
feature = feature * gen_flag_t
feature = tf.reshape(feature, [batch_size, time * seq_len, feature_out_dim / seq_len])
# batch_size * (time * sample_len) * dim

print("f", feature.shape)
print("gft", gen_flag_t.shape)