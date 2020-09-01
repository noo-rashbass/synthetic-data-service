import tensorflow as tf
from tensorflow.keras import layers
from output import OutputType, Normalization, Output
from util import *
from load_data import *
import numpy as np
from enum import Enum
import os

def make_discriminator(seq_len, features_dim, attributes_dim, num_layers=5, num_units=200):
    """
    discriminator model to distinguish real and fake samples

    Args:
    seq_len: maximum sequence length for input time series 
    features_dim: number of features after passing through "add_gen_flag" function
    attributes_dim: number of attributes after passing through "normalize_per_sample" function

    Returns:
    discriminator model
    """

    #define input shapes
    # functional API for multiple inputs
    input_feature_d = tf.keras.Input(shape=(seq_len, features_dim)) 
    input_attribute_d = tf.keras.Input(shape=(attributes_dim))
    input_feature_fl_d = tf.keras.layers.Flatten()(input_feature_d)
    input_attribute_fl_d = tf.keras.layers.Flatten()(input_attribute_d)

    # forward pass
    output_d = tf.keras.layers.Concatenate(axis=1)([input_feature_fl_d, input_attribute_fl_d])
    for i in range(num_layers - 1):
        output_d = tf.keras.layers.Dense(num_units, activation='relu')(output_d)
    output_d = tf.keras.layers.Dense(1)(output_d)
    output_d = tf.squeeze(output_d, axis=1)

    discriminator_model = tf.keras.models.Model(inputs=[input_feature_d, input_attribute_d], outputs=output_d)

    return discriminator_model

def make_attrdiscriminator(attributes_dim, num_layers=5, num_units=200):
    """
    attribute discriminator model to distinguish attributes in real and fake samples

    Args:
    features_dim: number of features after passing through "add_gen_flag" function

    Returns:
    discriminator model
    """

    #define input shape
    input_attribute_ad = tf.keras.Input(shape=(attributes_dim))

    # forward pass
    output_ad = tf.keras.layers.Flatten()(input_attribute_ad)
    for i in range(num_layers - 1):
        output_ad = tf.keras.layers.Dense(num_units, activation='relu')(output_ad)
    output_ad = tf.keras.layers.Dense(1)(output_ad)
    output_ad = tf.squeeze(output_ad, axis=1)

    attrdiscriminator_model = tf.keras.models.Model(inputs=input_attribute_ad, outputs=output_ad)

    return attrdiscriminator_model