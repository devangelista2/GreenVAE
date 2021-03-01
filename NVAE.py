import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.initializers import Zeros, RandomNormal
from tensorflow.keras.optimizers import Adam

from utils import load_data, utils, evaluate
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
            layers.Activation('sigmoid')]

    def call(self, inputs):
        h = inputs
        for layer in self.main_layers:
            h = layer(h)
        return layers.Multiply()([inputs, h])


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
            SE(self.c)]

    def call(self, inputs):
        h = inputs
        for layer in self.main_layers:
            h = layer(h)
        return layers.Add()([inputs, h])


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
            SE(self.c)]

    def call(self, inputs):
        h = inputs
        for layer in self.main_layers:
            h = layer(h)
        return layers.Add()([inputs, h])


class EncoderBlock(layers.Layer):
    def __init__(self, c, **kwargs):
        super().__init__(**kwargs)
        self.c = c
        self.main_layers = [
            EncoderCell(self.c),
            layers.Conv2D(self.c, 3, strides=1, padding='same', activation=activation, kernel_initializer=initializer)]

    def call(self, inputs):
        h = inputs
        for layer in self.main_layers:
            h = layer(h)
        return h


class DecoderBlock(layers.Layer):
    def __init__(self, c, **kwargs):
        super().__init__(**kwargs)
        self.c = c
        self.main_layers = [
            layers.Conv2D(self.c, 3, strides=1, padding='same', activation='relu', kernel_initializer=initializer),
            DecoderCell(self.c),
            layers.Conv2D(self.c, 3, strides=1, padding='same', activation='relu', kernel_initializer=initializer)]

    def call(self, inputs):
        h = inputs
        for layer in self.main_layers:
            h = layer(h)
        return h


class GetStatistics(layers.Layer):
    def __init__(self, z_dim, **kwargs):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.main_layers = [
            layers.Conv2D(self.z_dim, 3, strides=1, padding='same'),
            layers.Conv2D(self.z_dim, 3, strides=1, padding='same')]

    def call(self, inputs):
        z_mean = self.main_layers[0](inputs)
        z_log_var = self.main_layers[1](inputs)
        return z_mean, z_log_var


def get_nvae(x, z_dim, n_group_per_scale, n_ch=32, k_size=3):
    # Initialize the variables we will need
    bottom_up = []
    z_vec = []
    z_means = []
    z_log_vars = []
    delta_means = []
    delta_log_vars = []

    # Set the number of channels to be equal to initial_ch
    h = layers.Conv2D(n_ch, k_size, 1, padding='same', kernel_initializer=initializer, name='Initial_Conv2D')(x)

    # Bottom-up phase Encoder
    for idx, n_group in enumerate(n_group_per_scale):
        n_ch *= 2

        h = layers.Conv2D(n_ch, k_size, 2, padding='same', kernel_initializer=initializer, name='Conv2D_' + str(idx))(h)
        for group in range(n_group):
            h = EncoderBlock(K.int_shape(h)[-1], name='EncoderBlock_' + str(idx) + '_' + str(group))(h)
            bottom_up.append(h)

    # Top-down phase Decoder
    delta_mean, delta_log_var = GetStatistics(z_dim, name='initial_delta_stats')(bottom_up.pop(-1))
    z = layers.Lambda(utils.sampling)([delta_mean, delta_log_var])

    # Append everything
    z_vec.append(z)
    z_means.append([])
    z_log_vars.append([])
    delta_means.append(delta_mean)
    delta_log_vars.append(delta_log_var)

    for idx, n_group in enumerate(n_group_per_scale[::-1]):
        if idx == 0:
            start = 1
            n_channels = K.int_shape(h)[-1]
            h = z
        else:
            start = 0
            n_channels = K.int_shape(h)[-1]

        for group in range(start, n_group):
            h = DecoderBlock(n_channels, name='DecoderBlock_' + str(idx) + '_' + str(group))(h)
            h_z = layers.Concatenate()([h, bottom_up.pop(-1)])

            n_channels = K.int_shape(h)[-1]

            z_mean, z_log_var = GetStatistics(z_dim, name='stats_' + str(idx) + '_' + str(group))(h)
            delta_mean, delta_log_var = GetStatistics(z_dim, name='delta_stats_' + str(idx) + '_' + str(group))(h_z)
            z = layers.Lambda(utils.sampling)([z_mean + delta_mean, z_log_var + delta_log_var])

            # Append everything
            z_vec.append(z)
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)
            delta_means.append(delta_mean)
            delta_log_vars.append(delta_log_var)

            h = layers.Concatenate()([h, z])

        n_ch /= 2
        h = layers.Conv2DTranspose(n_ch, k_size, 2, padding='same', activation=activation,
                                   kernel_initializer=initializer, name='Conv2DT_' + str(idx))(h)

    h = DecoderBlock(n_ch, name='DecoderBlock_' + str(idx))(h)
    x_recon = layers.Conv2DTranspose(3, k_size, strides=1, padding='same', activation='sigmoid',
                                     kernel_initializer=initializer, name='Output')(h)

    # Define the two models (Encoder and NVAE)
    encoder = models.Model(x, [z_vec, delta_means, delta_log_vars])
    nvae = models.Model(x, x_recon)

    # Add the loss to the model
    nvae.add_loss(nvae_loss(x, x_recon, delta_means, delta_log_vars, z_means, z_log_vars))

    return nvae, encoder


def Decoder(latent_shape, n_group_per_scale, n_ch, k_size=3):
    z_dim = latent_shape[-1]

    z_in = layers.Input(shape=latent_shape)
    h = z_in

    for idx, n_group in enumerate(n_group_per_scale[::-1]):
        if idx == 0:
            start = 1
        else:
            start = 0
        n_channels = n_ch

        for group in range(start, n_group):
            h = DecoderBlock(n_channels, name='DecoderBlock_' + str(idx) + '_' + str(group))(h)

            n_channels = K.int_shape(h)[-1]

            z_mean, z_log_var = GetStatistics(z_dim, name='stats_' + str(idx) + '_' + str(group))(h)
            z = layers.Lambda(utils.sampling)([z_mean, z_log_var])

            h = layers.Concatenate()([h, z])

        n_ch /= 2
        h = layers.Conv2DTranspose(n_ch, k_size, 2, padding='same', activation=activation,
                                   kernel_initializer=initializer, name='Conv2DT_' + str(idx))(h)

    h = DecoderBlock(n_ch, name='DecoderBlock_' + str(idx))(h)
    x_recon = layers.Conv2DTranspose(3, k_size, strides=1, padding='same', activation='sigmoid',
                                     kernel_initializer=initializer, name='Output')(h)

    decoder = models.Model(z_in, x_recon)
    return decoder


def copy_weights(from_model, to_model):
    from_layers = from_model.layers
    to_layers = to_model.layers

    from_layers_name = [layer.name for layer in from_layers]

    for layer in to_layers:
        if layer.name in from_layers_name:
            print(layer.name)
            layer.set_weights(from_model.get_layer(layer.name).get_weights())
    return decoder


def nvae_loss(x_true, x_pred, delta_means, delta_log_vars, z_means, z_log_vars):
    L = len(delta_means)

    # Reconstruction loss: Mean squared error between x_true and x_pred
    L_rec = utils.recon(x_true, x_pred)

    # KL loss
    L_KL = 0
    for i in range(L):
        delta_mean = delta_means[i]
        delta_log_var = delta_log_vars[i]

        delta_mean = K.reshape(delta_mean, (-1, np.prod(K.int_shape(delta_mean)[1:])))
        delta_log_var = K.reshape(delta_log_var, (-1, np.prod(K.int_shape(delta_log_var)[1:])))

        if i == 0:
            L_KL = K.sum(K.square(delta_mean) + K.exp(delta_log_var) - 1 - delta_log_var, axis=-1)
        else:
            z_log_var = z_log_vars[i]
            z_log_var = K.reshape(z_log_var, (-1, np.prod(K.int_shape(z_log_var)[1:])))

            L_KL += K.sum(K.square(delta_mean) / K.exp(z_log_var) + K.exp(delta_log_var) - 1 - delta_log_var, axis=-1)
        L_KL = 0.5 * K.mean(L_KL)

    return L_rec + gamma * L_KL


# Setup the model by defining an activation function and an initializer
activation = 'swish'
initializer = RandomNormal(1e-2)

# Load data
data = "cifar10"  # or celeba
if data == "cifar10":
    x_train, x_test = load_data.load_cifar10()
elif data == "celeba":
    x_train, x_test = load_data.load_celeba()
else:
    print('Dataset not found.')

# Define hyperparameters
input_shape = x_train[0].shape
z_dim = 10

n_group_per_scale = [4, 2]
n_ch = 100
latent_shape = (input_shape[0] // (2 ** len(n_group_per_scale)), input_shape[1] // (2 ** len(n_group_per_scale)), z_dim)

gamma = 1e-2

epochs = 200
batch_size = 100

# Build the model
x = layers.Input(shape=input_shape)
nvae, encoder = get_nvae(x, z_dim, n_group_per_scale, n_ch)
decoder = Decoder(latent_shape, n_group_per_scale, n_ch * (2 ** len(n_group_per_scale)), k_size=3)

# Compile model
optimizer = utils.get_optimizer(x_train.shape[0] // batch_size)
nvae.compile(optimizer=optimizer, loss=None, metrics=['mse'])

# Fit model
hist = nvae.fit(x_train, None, batch_size=batch_size, epochs=epochs, verbose=1)
nvae.save_weights('saved_weights/NVAE_' + data + '.h5')

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
    second_vae.save_weights('saved_weights/secondstage_NVAE_' + data + '.h5')

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
    x_recon = nvae.predict(x_train)
    print('We lost ', evaluate.loss_variance(x_test, x_recon), 'Variance of the original data')

SHOW_GEN_FID = False
if SHOW_GEN_FID:
    if SECOND_STAGE:
        u_sample = np.random.normal(0, 1, (10000, latent_dim))
        z = second_decoder.predict(u_sample)
        z = np.reshape(z, (-1,) + latent_shape)
    elif GMM:
        z = utils.sample_from_GMM(z_density, 10000)
        z = np.reshape(z, (-1,) + latent_shape)
    else:
        z = np.random.normal(0, 1, (10000,) + latent_shape)
    x_gen = decoder.predict(z)

    gen_fid = evaluate.get_fid(x_test[:10000], x_gen)
    print('GEN FID:', gen_fid)

SHOW_REC_FID = False
if SHOW_REC_FID:
    x_recon = nvae.predict(x_train[:10000])

    rec_fid = evaluate.get_fid(x_test[:10000], x_recon)
    print('REC FID:', rec_fid)