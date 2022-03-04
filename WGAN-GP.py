import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K

from keras.callbacks import ModelCheckpoint

import keras.initializers

from functools import partial

import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt

