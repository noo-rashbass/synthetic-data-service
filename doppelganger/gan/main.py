from load_data import *
from util import *
import tensorflow as tf
import numpy as np
import os
import argparse

from network import make_discriminator, make_attrdiscriminator
from networkGenerator import DoppelGANgerGenerator
from doppelganger import DoppelGANger

def train(args):

    seq_len = args.seq_len
    batch_size = args.batch_size
    epochs = args.epochs
    total_generate_num_sample = args.total_generate_num_sample
    path_to_data = args.path_to_data
    output_path = args.output_path

    (data_feature, data_attribute, data_gen_flag, data_feature_outputs, data_attribute_outputs) = load_data(path_to_data)

    print("-----DATA LOADING-----")
    print(data_feature.shape)          # original features_dim        
    print(data_attribute.shape)        # original attributes_dim
    print(data_gen_flag.shape)
    num_real_attribute = len(data_attribute_outputs)

    (data_feature, data_attribute, data_attribute_outputs, real_attribute_mask) = \
        normalize_per_sample(data_feature, data_attribute, data_feature_outputs,data_attribute_outputs)

    print("-----DATA NORMALIZATION-----")
    print(real_attribute_mask)
    print(data_feature.shape)
    attributes_dim = data_attribute.shape[1]    # attributes_dim to be fed into model
    print(data_attribute.shape)       
    print(len(data_attribute_outputs))

    print("-----ADD GEN FLAG -----")
    data_feature, data_feature_outputs = add_gen_flag(
            data_feature, data_gen_flag, data_feature_outputs, seq_len)
    features_dim = data_feature.shape[2]    # features dim to be fed into model
    print(data_feature.shape)        
    print(len(data_feature_outputs))

    discriminator_model = make_discriminator(seq_len, features_dim, attributes_dim)
    attrdiscriminator_model = make_attrdiscriminator(attributes_dim)

    generator = DoppelGANgerGenerator(
            feed_back=False,
            noise=True,
            feature_outputs=data_feature_outputs,
            attribute_outputs=data_attribute_outputs,
            real_attribute_mask=real_attribute_mask,
            sample_len=seq_len)


    gan = DoppelGANger(
        epoch=epochs, 
        batch_size=batch_size, 
        data_feature=data_feature, 
        data_attribute=data_attribute, 
        real_attribute_mask=real_attribute_mask, 
        data_gen_flag=data_gen_flag,
        seq_len=seq_len, 
        data_feature_outputs=data_feature_outputs, 
        data_attribute_outputs=data_feature_outputs,
        generator = generator, 
        discriminator = discriminator_model, 
        d_rounds=1, 
        g_rounds=1, 
        d_gp_coe=10.,
        num_packing=1,
        attr_discriminator=attrdiscriminator_model,
        attr_d_gp_coe=10., 
        g_attr_d_coe=1.0,
        cumsum=False)

    #combine data attributes and features into one to be fed into the model
    # data_attribute_in = tf.expand_dims(data_attribute, axis=1)
    # data_attribute_in = tf.repeat(data_attribute_in, repeats=seq_len, axis=1)
    # data_all_in = tf.cast(tf.concat([data_feature, data_attribute_in], axis=2), dtype=tf.float32)

    print("----START TRAINING-----")
    # gan.compile()

    # if any callbacks are needed
    # callback1 = tf.keras.callbacks.EarlyStopping(monitor='d_loss', patience=3)
    # callback2 = tf.keras.callbacks.EarlyStopping(monitor='ad_loss', patience=3)
    # callback3 = tf.keras.callbacks.EarlyStopping(monitor='g_loss', patience=3)

    #gan.fit(data_all_in, batch_size=batch_size, epochs=epochs) #, callbacks=[callback1, callback2]
    gan.train_step()

    print("----FINISHED TRAINING-----")

    print("----START GENERATING------")

    if data_feature.shape[1] % seq_len != 0:
        raise Exception("length must be a multiple of sample_len")
    length = int(data_feature.shape[1] / seq_len)
    real_attribute_input_noise = gan.gen_attribute_input_noise(total_generate_num_sample) #(?,5)
    addi_attribute_input_noise = gan.gen_attribute_input_noise(total_generate_num_sample) #(?,5)
    feature_input_noise = gan.gen_feature_input_noise(total_generate_num_sample, length) #(?,1,5)
    input_data = gan.gen_feature_input_data_free(total_generate_num_sample) #(?,28)

    features, attributes, gen_flags, lengths = \
        gan.sample_from(real_attribute_input_noise, addi_attribute_input_noise,feature_input_noise, input_data)
    # specify given_attribute parameter, if you want to generate
    # data according to an attribute
    print("----SAMPLE FROM-----")
    print(features.shape)
    print(attributes.shape)
    print(gen_flags.shape)
    print(lengths.shape)

    features, attributes = renormalize_per_sample(features, attributes, data_feature_outputs,
        data_attribute_outputs, gen_flags, num_real_attribute=num_real_attribute)
    print("----RENORMALIZATION-----")
    print(features.shape)
    print(attributes.shape)

    np.savez(
            output_path,
            data_feature=features,
            data_attribute=attributes,
            data_gen_flag=gen_flags)

    print("Done")

if __name__ == "__main__":

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seq_len',
        default=130,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=64,
        type=int)
    parser.add_argument(
        '--epochs',
        help='Training epochs (should be optimized)',
        default=2,
        type=int)
    parser.add_argument(
        '--total_generate_num_sample',
        help='total number of samples to generate',
        default=1347,
        type=int)
    parser.add_argument(
        '--path_to_data',
        help='path to the pkl and npz files',
        default='data',
        type=str)
    parser.add_argument(
        '--output_path',
        help='path the generated file should be saved',
        default="generated_data_train.npz",
        type=str)
    args = parser.parse_args()

    train(args)