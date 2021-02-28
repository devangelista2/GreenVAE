import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal, GlorotUniform
from tensorflow.keras.optimizers import Adam
from utils import load_data, utils, evaluate
import two_stage


initializer = GlorotUniform() #RandomNormal(stddev=1e-2)
reg = None #'l2'
act = 'swish' #'relu'

"""
CUSTOM LAYERS
"""

class Film(layers.Layer):  #Feature-wise linear modulation
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.layers = [
          layers.Dense(self.c),
          layers.Dense(self.c),
          layers.Multiply(),
          layers.Add()
    ]

  def call(self, inputs):
    X,X_aux = inputs
    w,h = K.int_shape(X)[1], K.int_shape(X)[2]
    Dense_gamma,Dense_beta,mult,add = self.layers

    Gamma = layers.Reshape((w, h, self.c))(layers.RepeatVector(w*h)(Dense_gamma(X_aux)))
    Beta = layers.Reshape((w, h, self.c))(layers.RepeatVector(w*h)(Dense_beta(X_aux)))
    X = add([mult([X, Gamma]),Beta])
    return X

class SE(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.red_c = max(self.c // 16, int(self.c ** 0.5))
    self.layers = [
          layers.GlobalAveragePooling2D(),
          layers.Dense(self.red_c),
          layers.ReLU(),
          layers.Dense(self.c),
          layers.Activation('sigmoid')
    ]

  def call(self, inputs):
    X = inputs
    w, h = K.int_shape(X)[1], K.int_shape(X)[2]
    H = X
    for layer in self.layers:
      H = layer(H)
    H = layers.Reshape((w, h, self.c))(layers.RepeatVector(w * h)(H))
    return layers.Multiply()([X, H])


class EncoderCell(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.layers = [
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Conv2D(self.c, kernel_size=3, strides=1, padding='same',
                      kernel_regularizer=reg, kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Conv2D(self.c, kernel_size=3, strides=1, padding='same',
                      kernel_regularizer=reg,kernel_initializer=initializer),
        SE(self.c)
    ]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.layers:
      H = layer(H)
    return layers.Add()([X, H])


class DecoderCell(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.e_c = c * 6
    self.layers = [
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.DepthwiseConv2D(kernel_size=5, strides=1, padding='same',
                               kernel_regularizer=reg, kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Activation(act),
        layers.Conv2D(self.c, kernel_size=1, strides=1, padding='same',
                      kernel_regularizer=reg, kernel_initializer=initializer),
        SE(self.c)
    ]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.layers:
      H = layer(H)
    return layers.Add()([X, H])


class EncoderBlock(layers.Layer):
  def __init__(self, n_cell, n_conv, c, **kwargs):
    super().__init__(**kwargs)
    self.n_cell = n_cell
    self.n_conv = n_conv
    self.c = c
    self.layers = [EncoderCell(self.c) for _ in range(self.n_cell)]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.layers:
      H = layer(H)
    return H

class DecoderBlock(layers.Layer):
  def __init__(self, n_cell, n_conv, c, **kwargs):
    super().__init__(**kwargs)
    self.n_cell = n_cell
    self.n_conv = n_conv
    self.c = c
    self.layers = [DecoderCell(self.c) for _ in range(self.n_cell)]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.layers:
      H = layer(H)
    return H

def HFVAE(x, z_dim, n_group_per_scale, initial_ch=32, k_size=3):

  # Bottom-up phase encoder
  h = layers.Conv2D(initial_ch, kernel_size=1, strides=1, padding='same',
                    kernel_regularizer=reg, kernel_initializer=initializer, name='Conv2D_0')(x)
  n_ch = initial_ch
  levels = []

  for g,n_group in enumerate(n_group_per_scale):
    n_ch *= 2

    h = layers.Conv2D(n_ch, kernel_size=k_size, strides=2, padding='same',
                      kernel_regularizer=reg, kernel_initializer=initializer, name='Conv2D_s2_'+str(g))(h)
    for group in range(n_group):
      locname = str(g) + "_" + str(group)
      h = EncoderBlock(1, 1, n_ch, name='EncoderBlock_'+locname)(h)
      levels.append(h)

  levels.reverse()
  n_group_per_scale.reverse()

  # Top-down phase encoder

  latent_variables = list()
  latent_stats = list()
  delta_stats = list()

  stats = layers.GlobalAveragePooling2D()(levels[0])
  delta_mean = layers.Dense(64,name='dense_delta_mean_0')(stats)
  delta_log_var = layers.Dense(64, name='dense_delta_log_var_0')(stats)

  z = layers.Lambda(sampling)([delta_mean, delta_log_var])
  latent_variables.append(z)
  latent_stats.append(())
  delta_stats.append((delta_mean, delta_log_var))
  dim = levels[0].shape[1]
  z = layers.Dense((dim*dim*n_ch),name='dense_z_orig')(z)
  h = layers.Reshape((dim,dim,n_ch))(z)

  level = 1
  first = True

  for g,n_group in enumerate(n_group_per_scale):
    if first:
        start = 1
        first = False
    else:
        start = 0
    for group in range(start, n_group):
      #print(n_group,group,h.shape)
      locname = str(g)+"_"+str(group)

      h = DecoderBlock(1, 1, n_ch, name='DecoderBlock_'+locname)(h)

      h_z = layers.Concatenate()([h, levels[level]])
      #print("shapes = ", h.shape, levels[level].shape)
      level = level + 1

      z_stats= layers.GlobalAveragePooling2D()(h)
      d_stats = layers.GlobalAveragePooling2D()(h_z)

      z_mean = layers.Dense(z_dim,name='dense_z_mean'+locname)(z_stats)
      z_log_var = layers.Dense(z_dim,name='dense_z_log_var'+locname)(z_stats)
      delta_mean = layers.Dense(z_dim,name='dense_delta_mean'+locname)(d_stats)
      delta_log_var = layers.Dense(z_dim,name='dense_delta_log_var'+locname)(d_stats)

      z = layers.Lambda(sampling)([z_mean + delta_mean, z_log_var + delta_log_var])
      latent_variables.append(z)
      latent_stats.append((z_mean, z_log_var))
      delta_stats.append((delta_mean, delta_log_var))

      h = Film((K.int_shape(h)[3]), name='Film' + locname)([h, z])
      h = layers.Conv2D(n_ch, kernel_size=k_size, strides=1, padding='same', kernel_regularizer=reg,
                        kernel_initializer=initializer, name='Conv2D_1s_'+locname)(h)

    dim = dim*2
    n_ch = n_ch//2
    h = layers.Conv2DTranspose(n_ch, k_size, strides=2, padding='same', activation=act, kernel_regularizer=reg,
                               kernel_initializer=initializer, name='Conv2DT_2s_'+str(g))(h)

  x_recon = layers.Conv2DTranspose(3, k_size, strides=1, padding='same', activation='sigmoid',
                                   kernel_initializer=initializer, name='Output')(h)

  encoder = models.Model(x, [latent_variables,delta_stats,latent_stats])
  hfvae = models.Model(x, x_recon)

  # adding loss

  x_true = K.reshape(x, (-1, np.prod(input_shape)))
  x_pred = K.reshape(x_recon, (-1, np.prod(input_shape)))
  L_recon = K.sum(K.square(x_true - x_pred), axis=-1)

  hfvae.add_loss(L_recon)

  for i in range(len(latent_stats)):
    delta_mean, delta_log_var = delta_stats[i]
    if i == 0:
        hfvae.add_loss(0.5 * gamma * K.sum(K.square(delta_mean) + K.exp(delta_log_var) - 1 - delta_log_var, axis=-1))
    else:
        z_mean, z_log_var = latent_stats[i]
        hfvae.add_loss(0.5 * gamma * K.sum(K.square(delta_mean) / K.exp(z_log_var) + K.exp(delta_log_var) - 1 - delta_log_var, axis=-1))

  return hfvae, encoder, latent_variables, latent_stats, delta_stats


######################################## DECODER ##########################################################

def Decoder(z_dim, n_group_per_scale, n_ch, k_size=3):
  dim = 32//(2**len(n_group_per_scale))
  z_in = layers.Input(shape=64)
  n_ch = n_ch*2**len(n_group_per_scale)
  #print(dim,n_ch)

  z = layers.Dense(dim * dim * n_ch,name='dense_z_orig')(z_in)
  h = layers.Reshape((dim, dim, n_ch))(z)

  first = True

  for g,n_group in enumerate(n_group_per_scale):
    if first:
        start = 1
        first = False
    else:
        start = 0
    for group in range(start, n_group):
      locname = str(g)+"_"+str(group)
      h = DecoderBlock(1, 1, n_ch, name='DecoderBlock_'+locname)(h)

      z_stats = layers.GlobalAveragePooling2D()(h)
      z_mean = layers.Dense(z_dim, name='dense_z_mean' + locname)(z_stats)
      z_log_var = layers.Dense(z_dim, name='dense_z_log_var' + locname)(z_stats)

      z = layers.Lambda(sampling)([z_mean, z_log_var])

      h = Film(K.int_shape(h)[3], name='Film' + locname)([h, z])
      h = layers.Conv2D(n_ch, kernel_size=k_size, strides=1, padding='same', kernel_regularizer=reg,
                        kernel_initializer=initializer, name='Conv2D_1s_' + locname)(h)

    dim = dim * 2
    n_ch = n_ch // 2
    h = layers.Conv2DTranspose(n_ch, k_size, strides=2, padding='same', activation=act, kernel_regularizer=reg,
                               kernel_initializer=initializer, name='Conv2DT_2s_'+str(g))(h)

  x_gen = layers.Conv2DTranspose(3, k_size, strides=1, padding='same', activation='sigmoid',
                                 kernel_regularizer=reg, kernel_initializer=initializer, name='Output')(h)
  decoder = models.Model(z_in, x_gen)

  return decoder

#################################### FUNCTIONS ################################################

def sampling(args):
  z_mean, z_log_var = args
  return z_mean + K.random_normal(K.shape(z_mean), 0, 1) * K.exp(0.5 * z_log_var)


def copy_weights(hvae, decoder):
    hvae_layers = hvae.layers
    decoder_layers = decoder.layers

    hvae_layers_name = []
    for layer in hvae_layers:
      hvae_layers_name.append(layer.name)

    for layer in decoder_layers:
      if layer.name in hvae_layers_name:
        print(layer.name)
        layer.set_weights(hvae.get_layer(layer.name).get_weights())
    return decoder

########################################### TRAINING  ##########################################################
input_shape = (32, 32, 3)
z_dim = 12
n_group_per_scale = [1,1,1,1]
n_ch = 64 #128
k_size = 3
gamma = 0.01
x = layers.Input(shape=input_shape)
hfvae, encoder, latent_variables, latent_stats, delta_stats = HFVAE(x, z_dim, n_group_per_scale, n_ch, k_size)
decoder = Decoder(z_dim, n_group_per_scale, n_ch)

hfvae.summary()

hfvae.compile(optimizer=Adam(learning_rate=0.00001), loss=[], metrics=['mse'])

# Load data
x_train, x_test = load_data.load_cifar10()

weights_filename = 'hfvae_64.h5'

TRAIN = True

epochs = 0
batch_size = 100

if TRAIN:
    hfvae.load_weights('saved_weights/'+weights_filename)
    hfvae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=1)
    hfvae.save_weights('saved_weights/'+weights_filename)
else:
	hfvae.load_weights('saved_weights/'+weights_filename)

# Load Weights into the
decoder = copy_weights(hfvae, decoder)

SECOND_STAGE = False
if SECOND_STAGE:
  z_train = encoder.predict(x_train)[0][0]

  latent_dim = np.prod(z_train.shape[1:])
  z_train = np.reshape(z_train, (-1, latent_dim))

  second_vae, second_encoder, second_decoder = two_stage.get_second_stage(latent_dim)

  # Compile model
  optimizer = utils.get_optimizer(z_train.shape[0] // batch_size, initial_lr=1e-3)
  second_vae.compile(optimizer=optimizer, loss=None, metrics=[utils.cos_sim])

  second_vae.fit(z_train, None, batch_size=batch_size, epochs=epochs)
  second_vae.save_weights('saved_weights/secondstage_NVAE_' + data + '.h5')

GMM = True
if GMM:
  from sklearn.mixture import GaussianMixture

  #we may only work on z_mean of the innermost layer
  z_train = encoder.predict(x_train)[0][0]
  #print("ltatent dim = ",z_train.shape[1])

  z_density = GaussianMixture(n_components=10, max_iter=100)
  z_density.fit(z_train)

########################################################################################################################
# SHOW THE RESULTS
########################################################################################################################

SHOW_METRICS = True
if SHOW_METRICS:
    _, delta_stats, latent_stats = encoder.predict(x_test, batch_size=100)

    for i in range(len(delta_stats)):
        threshold = .9
        if i == 0:
            z_mean, z_var = 0, 1
        else:
            z_mean, z_log_var = latent_stats[i]
            z_var= np.mean(np.exp(z_log_var),axis=0)
        delta_mean, delta_log_var = delta_stats[i]
        delta_var = np.mean(np.exp(delta_log_var)/z_var,axis=0)
        print("inactive in group {}: {} out of {}".format(i, np.sum(delta_var > threshold), np.shape(delta_var)[0]))

SHOW_GEN_FID = True
if SHOW_GEN_FID:
  if SECOND_STAGE:
    u_sample = np.random.normal(0, 1, (10000, 64))
    z = second_decoder.predict(u_sample)
  elif GMM:
    z = utils.sample_from_GMM(z_density, 10000)
  else:
    z = np.random.normal(0, 1, (10000, 64))
  x_gen = decoder.predict(z)

  gen_fid = evaluate.get_fid(x_test[:10000], x_gen)
  print('GEN FID:', gen_fid)

SHOW_REC_FID = True
if SHOW_REC_FID:
  x_recon = hfvae.predict(x_train[:10000])

  rec_fid = evaluate.get_fid(x_test[:10000], x_recon)
  print('REC FID:', rec_fid)
