import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.initializers import Zeros, RandomNormal
from tensorflow.keras.optimizers import Adam

from utils import load_data, utils
import two_stage

class SE(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.red_c = max(self.c // 16, int(self.c ** 0.5))
    self.main_layers = [
          layers.GlobalAveragePooling2D(),
          layers.Dense(self.red_c),
          layers.ReLU(),
          layers.Dense(self.c),
          layers.Activation('sigmoid')
    ]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.main_layers:
      H = layer(H)
    return layers.Multiply()([X, H])


class EncoderCell(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.main_layers = [
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.Conv2D(self.c, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.Conv2D(self.c, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer),
        SE(self.c)
    ]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.main_layers:
      H = layer(H)
    return layers.Add()([X, H])


class DecoderCell(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.e_c = c * 6
    self.main_layers = [
        layers.BatchNormalization(),
        layers.Conv2D(self.e_c, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer),
        layers.BatchNormalization(), 
        layers.Activation(activation),
        layers.DepthwiseConv2D(kernel_size=5, strides=1, padding='same', kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Activation(activation),
        layers.Conv2D(self.c, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer), 
        layers.BatchNormalization(), 
        SE(self.c)
    ]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.main_layers:
      H = layer(H)
    return layers.Add()([X, H])


class EncoderBlock(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.main_layers = [
        EncoderCell(self.c),
        layers.Conv2D(self.c, 3, strides=1, padding='same', activation=activation, kernel_initializer=initializer)
    ]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.main_layers:
      H = layer(H)
    return H


class DecoderBlock(layers.Layer):
  def __init__(self, c, **kwargs):
    super().__init__(**kwargs)
    self.c = c
    self.main_layers = [
        DecoderCell(self.c),
        layers.Conv2D(self.c, 3, strides=1, padding='same', activation='relu', kernel_initializer=initializer)
    ]

  def call(self, inputs):
    X = inputs
    H = X
    for layer in self.main_layers:
      H = layer(H)
    return H

class GetStatistics(layers.Layer):
  def __init__(self, z_dim, **kwargs):
    super().__init__(**kwargs)
    self.z_dim = z_dim
    self.main_layers = [
        layers.Conv2D(self.z_dim, 3, strides=1, padding='same'),
        layers.Conv2D(self.z_dim, 3, strides=1, padding='same')
    ]

  def call(self, inputs):
    X = inputs
    z_mean = self.main_layers[0](X)
    z_log_var = self.main_layers[1](X)

    return z_mean, z_log_var

def NVAE(x, z_dim, n_group_per_scale, initial_ch=32, k_size=3):

  # Bottom-up phase encoder
  h = layers.Conv2D(initial_ch, kernel_size=k_size, strides=1, padding='same', kernel_initializer=initializer, name='Conv2D_'+str(1))(x)
  h = layers.BatchNormalization()(h)

  n_ch = initial_ch
  levels = []
  for n_group in n_group_per_scale:
    n_ch *= 2

    h = layers.Conv2D(n_ch, kernel_size=k_size, strides=2, padding='same', kernel_initializer=initializer, name='Conv2D_'+str(n_group))(h)
    for group in range(n_group):

      h = EncoderBlock(K.int_shape(h)[-1], name='EncoderBlock_'+str(n_group)+'_'+str(group))(h)
      levels.append(h)

  levels.reverse()

  # Top-down phase encoder
  latent_variables = list()
  latent_stats = list()
  delta_stats = list()

  delta_mean, delta_log_var = GetStatistics(z_dim, name='delta_stats_'+str(0))(levels[0])
  z = layers.Lambda(utils.sampling)([delta_mean, delta_log_var])

  latent_variables.append(z)
  latent_stats.append(())
  delta_stats.append((delta_mean, delta_log_var))

  h = layers.Conv2DTranspose(n_ch, k_size, activation=activation, padding='same', kernel_initializer=initializer, name='Conv2DT_'+str(1))(z)
  group = 0

  for n_group in n_group_per_scale:
    
    for group in range(1, n_group):
      h = DecoderBlock(K.int_shape(h)[-1], name='DecoderBlock_'+str(n_group)+'_'+str(group))(h)
      h_z = layers.Concatenate()([h, levels[group]])

      z_mean, z_log_var = GetStatistics(z_dim, name='stats_'+str(n_group)+'_'+str(group))(h)
      delta_mean, delta_log_var = GetStatistics(z_dim, name='delta_stats_'+str(n_group)+'_'+str(group))(h_z)

      z = layers.Lambda(utils.sampling)([z_mean + delta_mean, z_log_var + delta_log_var])

      latent_variables.append(z)
      latent_stats.append((z_mean, z_log_var))
      delta_stats.append((delta_mean, delta_log_var))

      h = layers.Concatenate()([h, z])

    n_ch /= 2
    h = layers.Conv2DTranspose(n_ch, k_size, strides=2, padding='same', activation=activation, kernel_initializer=initializer, name='Conv2DT_'+str(2))(h)

  h = DecoderBlock(K.int_shape(h)[-1], name='DecoderBlock_'+str(group+1))(h)
  x_recon = layers.Conv2DTranspose(3, k_size, strides=1, padding='same', activation='sigmoid', kernel_initializer=initializer, name='Output')(h)

  encoder = models.Model(x, [latent_variables, delta_stats])
  nvae = models.Model(x, x_recon)
  nvae.add_loss(nvae_loss(x, x_recon, delta_stats, latent_stats))
    
  return nvae, encoder, latent_variables, latent_stats, delta_stats


def Decoder(z_dim, n_group_per_scale, n_ch, k_size=3):
  z_in = layers.Input(shape=(16, 16, z_dim))
  h = layers.Conv2DTranspose(n_ch, k_size, activation=activation, padding='same', kernel_initializer=initializer, name='Conv2DT_'+str(1))(z_in)
  group = 0

  for n_group in n_group_per_scale:
    
    for group in range(1, n_group):
      h = DecoderBlock(K.int_shape(h)[-1], name='DecoderBlock_'+str(n_group)+'_'+str(group))(h)

      z_mean, z_log_var = GetStatistics(z_dim, name='stats_'+str(n_group)+'_'+str(group))(h)
      z = layers.Lambda(utils.sampling)([z_mean, z_log_var])

      h = layers.Concatenate()([h, z])

    n_ch /= 2
    h = layers.Conv2DTranspose(n_ch, k_size, strides=2, padding='same', activation=activation, kernel_initializer=initializer, name='Conv2DT_'+str(2))(h)

  h = DecoderBlock(K.int_shape(h)[-1], name='DecoderBlock_'+str(group+1))(h)
  x_gen = layers.Conv2DTranspose(3, k_size, strides=1, padding='same', activation='sigmoid', kernel_initializer=initializer, name='Output')(h)
  decoder = models.Model(z_in, x_gen)

  return decoder

def copy_weights(nvae, decoder):
	nvae_layers = nvae.layers
	decoder_layers = decoder.layers

	nvae_layers_name = []
	for layer in nvae_layers:
	  nvae_layers_name.append(layer.name)

	for layer in decoder_layers:
	  if layer.name in nvae_layers_name:
	    print(layer.name)
	    layer.set_weights(nvae.get_layer(layer.name).get_weights())
	return decoder


def nvae_loss(x_true, x_pred, delta_stats, latent_stats):

	L_rec = utils.recon(x_true, x_pred)
	L_KL = 0

	for i in range(len(latent_stats)):
		delta_mean, delta_log_var = delta_stats[i]

		latent_dim = K.int_shape(delta_mean)[1]

		delta_mean = K.reshape(delta_mean, (-1, latent_dim*latent_dim*z_dim))
		delta_log_var = K.reshape(delta_log_var, (-1, latent_dim*latent_dim*z_dim))

		if i==0:
			L_KL = K.sum(K.square(delta_mean) + K.exp(delta_log_var) - 1 - delta_log_var, axis=-1)
		else:
			z_mean, z_log_var = latent_stats[i]
			z_log_var = K.reshape(z_log_var, (-1, latent_dim*latent_dim*z_dim))
			L_KL += K.sum(K.square(delta_mean) / K.exp(z_log_var) + K.exp(delta_log_var) - 1 - delta_log_var, axis=-1)
		L_KL = 0.5 * K.mean(L_KL)

	return L_rec + gamma * L_KL


# Setup the model by defining an activation function and an initializer
activation = 'swish'
initializer=RandomNormal(1e-2)

# Load data
data = "cifar10" # or celeba
if data == "cifar10":
    x_train, x_test = load_data.load_cifar10()
elif data == "celeba":
    x_train, x_test = load_data.load_celeba()
else:
    print('Dataset not found.')

# Define hyperparameters
input_shape = x_train[0].shape
z_dim = 10

n_group_per_scale = [4]
n_ch = 100

gamma = 1e-2

epochs = 200
batch_size = 100


# Build the model
x = layers.Input(shape=input_shape)
nvae, encoder, latent_variables, latent_stats, delta_stats = NVAE(x, z_dim, n_group_per_scale, n_ch)
decoder = Decoder(z_dim, n_group_per_scale, n_ch)

# Compile model
optimizer = utils.get_optimizer(x_train.shape[0] // batch_size)
nvae.compile(optimizer=optimizer, loss=None, metrics=['mse'])

# Fit model
hist = nvae.fit(x_train, None, batch_size=batch_size, epochs=epochs, verbose=1)
nvae.save_weights('saved_weights/NVAE_' + data +'.h5')

# Load Weights into the Decoder
decoder = copy_weights(nvae, decoder)


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
    second_vae.save_weights('saved_weights/secondstage_NVAE_' + data +'.h5')

GMM = False
if GMM:
    from sklearn.mixture import GaussianMixture

    z_train = encoder.predict(x_train)[0][0]
    
    latent_dim = np.prod(z_train.shape[1:])
    z_train = np.reshape(z_train, (-1, latent_dim))

    z_density = GaussianMixture(n_components=10, max_iter=100)
    z_density.fit(z_train)

########################################################################################################################
# SHOW THE RESULTS
########################################################################################################################

SHOW_METRICS = False
if SHOW_METRICS:
    x_recon = vae.predict(x_train)
    print('We lost ', evaluate.loss_variance(x_test, x_recon), 'Variance of the original data')

SHOW_GEN_FID = False
if SHOW_GEN_FID:
    if SECOND_STAGE:
        u_sample = np.random.normal(0, 1, (10000, latent_dim))
        z = second_decoder.predict(u_sample)
        z = np.reshape(z, (-1, 16, 16, z_dim))
    elif GMM:
        z = utils.sample_from_GMM(z_density, 10000)
        z = np.reshape(z, (-1, 16, 16, z_dim))
    else:
        z = np.random.normal(0, 1, (10000, 16, 16, z_dim))
    x_gen = decoder.predict(z)

    gen_fid = evaluate.get_fid(x_test[:10000], x_gen)
    print('GEN FID:', gen_fid)

SHOW_REC_FID = False
if SHOW_REC_FID:
    x_recon = vae.predict(x_train[:10000])

    rec_fid = evaluate.get_fid(x_test[:10000], x_recon)
    print('REC FID:', rec_fid)





