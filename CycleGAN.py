from tensorflow_addons.layers import InstanceNormalization

from keras.layers import Input, Dropout, Concatenate
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
#from models.layers.layers import ReflectionPadding2D
from keras.models import Model
#from keras.initializers import RandomNormal
#from keras.optimizers import Adam

#from keras.utils import plot_model

import numpy as np
from utils.loaders import DataLoader

data_loader = DataLoader(dataset_name='appora', img_res=(128,128))
input_dim=(128,128,3)
learning_rate=0.0002
lambda_validation=1
lambda_reconstr=10
lambda_id=2
generator_type='u-net'
gen_n_filters=32
disc_n_filters=32

def build_gen():
    def downsample(layer_input, filters, f_size=4):
        d=Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d=InstanceNormalization(axis=-1, center=False, scale=False)(d)
        d=Activation('relu')(d)
        return d

    def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        u=UpSampling2D(size=2)(layer_input)
        u=Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
        u=InstanceNormalization(axis=-1, center=False, scale=False)(u)
        u=Activation('relu')(u)
        if dropout_rate:
            u=Dropout(dropout_rate)(u)

        u=Concatenate()([u,skip_input])
        return u
    img=Input(shape=input_dim)

    #downsampling
    d1 = downsample(img,gen_n_filters)
    d2 = downsample(d1, gen_n_filters*2)
    d3 = downsample(d2, gen_n_filters*4)
    d4 = downsample(d3, gen_n_filters*8)

    #upsample
    u1=upsample(d4,d3,gen_n_filters*4)
    u2=upsample(u1,d2,gen_n_filters*2)
    u3=upsample(u2,d1,gen_n_filters)

    u4=UpSampling2D(size=2)(u3)
    output=Conv2D(3,kernel_size=4,strides=1,padding='same',activation='tanh')(u4)
    return Model(img,output)

model=build_gen()
model.summary()

def build_dis():
    def conv4(layer_input, filters, stride=2, norm=True):
        y=Conv2D(filters, kernel_size=4, strides=stride, padding='same')(layer_input)
        if norm:
            y=InstanceNormalization(axis=-1, center=False, scale=False)(y)
        y=LeakyReLU(0.2)(y)
        return y

    img=Input(shape=input_dim)
    y=conv4(img, disc_n_filters, stride=2, norm=False)
    y = conv4(y, disc_n_filters * 2, stride=2)
    y = conv4(y, disc_n_filters * 4, stride=2)
    y = conv4(y, disc_n_filters * 8, stride=1)
    output=Conv2D(1,kernel_size=4, strides=1, padding='same')(y)
    return Model(img,output)

disc_model=build_dis()
disc_model.summary()

#compiling

d_A=build_dis()
d_B=build_dis()

d_A.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
d_B.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

g_AB=build_gen()
g_BA=build_gen()

d_A.trainable=False
d_B.trainable=False

img_A=Input(shape=input_dim)
img_B=Input(shape=input_dim)
fake_A=g_BA(img_B)
fake_B=g_AB(img_A)

valid_A=d_A(fake_A)
valid_B=d_B(fake_B)

reconstr_A=g_BA(fake_B)
reconstr_B=g_AB(fake_A)

img_A_id=g_BA(img_A)
img_B_id=g_AB(img_B)

combined=Model(inputs=[img_A,img_B], outputs=[valid_A,valid_B,reconstr_A,reconstr_B,img_A_id,img_B_id])
combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights=[lambda_validation,lambda_validation,lambda_reconstr,lambda_reconstr,lambda_id,lambda_id], optimizer='adam')
combined.summary()

img_rows=input_dim[1]
batch_size=1

patch=int(img_rows/2**3)
disc_patch=(patch, patch, 1)

valid=np.ones((batch_size,)+disc_patch)
fake=np.zeros((batch_size,)+disc_patch)

for epoch in range(0,20):
    for batch_i,(imgs_A,imgs_B) in enumerate(data_loader.load_batch(batch_size)):
        fake_B=g_AB.predict(imgs_A)
        fake_A=g_BA.predict(imgs_B)

        dA_loss_real = d_A.train_on_batch(imgs_A,valid)
        dA_loss_fake = d_A.train_on_batch(fake_A,fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = d_A.train_on_batch(imgs_B, valid)
        dB_loss_fake = d_A.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        d_loss = 0.5 * np.add(dA_loss,dB_loss)
        g_loss=combined.train_on_batch([imgs_A,imgs_B],[valid,valid, imgs_A, imgs_B, imgs_A, imgs_B])





