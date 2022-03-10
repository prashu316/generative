import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge
from tensorflow import keras
from keras.models import Model, Sequential
from keras import backend as K
#from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal

from functools import partial

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
  x = Dense(512, kernel_initializer = weight_init)(x)
  x = get_activation(critic_activation)(x)
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
  x = Reshape([7,7,64])(x)
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

def get_opti(self, lr):
  if self.optimiser == 'adam':
    opti = Adam(lr=lr, beta_1=0.5)
  elif self.optimiser == 'rmsprop':
    opti = RMSprop(lr=lr)
  else:
    opti = Adam(lr=lr)

  return opti

def set_trainable(self, m, val):
  m.trainable = val
  for l in m.layers:
    l.trainable = val

    def _build_adversarial():
        # Freeze generator's layers while training critic
        set_trainable(generator, False)

        # Image input (real sample)
        real_img = Input(shape=input_dim)

        # Fake image
        z_disc = Input(shape=(z_dim,))
        fake_img = generator(z_disc)

        # critic determines validity of the real and fake images
        fake = critic(fake_img)
        valid = critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'interpolated_samples' argument
        partial_gp_loss = partial(gradient_penalty_loss,
                                  interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        critic_model = Model(inputs=[real_img, z_disc],
                             outputs=[valid, fake, validity_interpolated])

        critic_model.compile(
            loss=[wasserstein, wasserstein, partial_gp_loss]
            , optimizer=get_opti(critic_learning_rate)
            , loss_weights=[1, 1, grad_weight]
        )

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        set_trainable(critic, False)
        set_trainable(generator, True)

        # Sampled noise for input to generator
        model_input = Input(shape=(z_dim,))
        # Generate images based of noise
        img = generator(model_input)
        # Discriminator determines validity
        model_output = critic(img)
        # Defines generator model
        model = Model(model_input, model_output)

        model.compile(optimizer=get_opti(generator_learning_rate)
                      , loss=wasserstein
                      )

        set_trainable(critic, True)
    critic_model, model=_build_adversarial()

    def train_critic(x_train, batch_size, using_generator):

        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32)  # Dummy gt for gradient penalty

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, z_dim))

        d_loss = critic_model.train_on_batch([true_imgs, noise], [valid, fake, dummy])
        return d_loss


    def train_generator(batch_size):
        valid = np.ones((batch_size,1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        return model.train_on_batch(noise, valid)

    epoch = 0

    def train(x_train, batch_size, epochs, print_every_n_batches=10
              , n_critic=5
              , using_generator=False):

        for epoch in range(0, epochs):

            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic

            for _ in range(critic_loops):
                d_loss = train_critic(x_train, batch_size, using_generator)

            g_loss = train_generator(batch_size)

            print("%d (%d, %d) [D loss: (%.1f)(R %.1f, F %.1f, G %.1f)] [G loss: %.1f]" % (
            epoch, critic_loops, 1, d_loss[0], d_loss[1], d_loss[2], d_loss[3], g_loss))

            sd_losses.append(d_loss)
            g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                model.save_weights('try1.h5')

            epoch += 1

train(x_train,64,100,6000,5)