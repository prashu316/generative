import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge
import keras.datasets
from keras.models import Model, Sequential
from keras import backend as K

from keras.callbacks import ModelCheckpoint

import keras.initializers

from functools import partial
from tensorflow import keras
import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt

input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, interpolated_samples):
    """
            Computes gradient penalty based on prediction and weighted real / fake samples
            """
    gradients = K.gradients(y_pred, interpolated_samples)[0]

    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

    def wasserstein(y_true, y_pred):
        return -K.mean(y_true * y_pred)


def get_activation(activation):
    if activation == 'leaky_relu':
        layer = LeakyReLU(alpha=0.2)
    else:
        layer = Activation(activation)
    return layer

#parameters
input_dim = (128,128,3)
critic_conv_filters = [64,128,256,512]
critic_conv_kernel_size = [5,5,5,5]
critic_conv_strides = [2,2,2,2]
critic_batch_norm_momentum = None
critic_activation = 'leaky_relu'
critic_dropout_rate = None
critic_learning_rate = 0.0002
generator_initial_dense_layer_size = (4, 4, 512)
generator_upsample = [1,1,1,1]
generator_conv_filters = [256,128,64,3]
generator_conv_kernel_size = [5,5,5,5]
generator_conv_strides = [2,2,2,2]
generator_batch_norm_momentum = 0.9
generator_activation = 'leaky_relu'
generator_dropout_rate = None
generator_learning_rate = 0.0002
optimiser = 'adam'
grad_weight = 10
z_dim = 100
batch_size = 1