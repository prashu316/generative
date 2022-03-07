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
input_dim = (28,28,1)
critic_conv_filters = [64,128,256,512]
critic_conv_kernel_size = [5,5,5,5]
critic_conv_strides = [2,2,2,2]
critic_batch_norm_momentum = None
critic_activation = 'leaky_relu'
critic_dropout_rate = None
critic_learning_rate = 0.0002
generator_initial_dense_layer_size = (8, 8, 512)
generator_upsample = [1,1,1,1]
generator_conv_filters = [256,128,64,1]
generator_conv_kernel_size = [3,3,3,3]
generator_conv_strides = [1,2,2,1]
generator_batch_norm_momentum = 0.9
generator_activation = 'leaky_relu'
generator_dropout_rate = None
generator_learning_rate = 0.0002
optimiser = 'adam'
grad_weight = 10
z_dim = 2
batch_size = 64
weight_init = RandomNormal(mean=0., stddev=0.02)
n_layers_critic=4
n_layers_generator=4

def _build_critic():
  critic_input = Input(shape=input_dim, name='critic_input')
  x = critic_input
  for i in range(n_layers_critic):
    x = Conv2D(
                filters = critic_conv_filters[i]
                , kernel_size = critic_conv_kernel_size[i]
                , strides = critic_conv_strides[i]
                , padding = 'same'
                , name = 'critic_conv_' + str(i)
                , kernel_initializer = weight_init
                )(x)
    if critic_batch_norm_momentum and i > 0:
      x = BatchNormalization(momentum = critic_batch_norm_momentum)(x)
    x = get_activation(critic_activation)(x)
    if critic_dropout_rate:
      x = Dropout(rate = critic_dropout_rate)(x)
  x = Flatten()(x)
  # x = Dense(512, kernel_initializer = weight_init)(x)
  # x = self.get_activation(self.critic_activation)(x)
  critic_output = Dense(1, activation=None
        , kernel_initializer = weight_init
        )(x)
  return Model(critic_input, critic_output)
critic=_build_critic()
critic.summary()

def _build_generator():
  generator_input = Input(shape=(z_dim,), name='generator_input')
  x = generator_input
  x = Dense(np.prod(generator_initial_dense_layer_size), kernel_initializer = weight_init)(x)
  if generator_batch_norm_momentum:
    x = BatchNormalization(momentum = generator_batch_norm_momentum)(x)
  x = get_activation(generator_activation)(x)
  x = Reshape(generator_initial_dense_layer_size)(x)
  if generator_dropout_rate:
    x = Dropout(rate = generator_dropout_rate)(x)
  for i in range(n_layers_generator):
    if generator_upsample[i] == 2:
      x = UpSampling2D()(x)
      x = Conv2D(
                filters = generator_conv_filters[i]
                , kernel_size = generator_conv_kernel_size[i]
                , padding = 'same'
                , name = 'generator_conv_' + str(i)
                , kernel_initializer = weight_init
                )(x)
    else:

      x = Conv2DTranspose(
                    filters = generator_conv_filters[i]
                    , kernel_size = generator_conv_kernel_size[i]
                    , padding = 'same'
                    , strides = generator_conv_strides[i]
                    , name = 'generator_conv_' + str(i)
                    , kernel_initializer = weight_init
                    )(x)

    if i < n_layers_generator - 1:
      if generator_batch_norm_momentum:
        x = BatchNormalization(momentum = generator_batch_norm_momentum)(x)
      x = get_activation(generator_activation)(x)
    else:
      x = Activation('tanh')(x)
  generator_output = x
  return Model(generator_input, generator_output)
generator=_build_generator()
generator.summary()

