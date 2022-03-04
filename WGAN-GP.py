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
