# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Dense, Flatten, Reshape, Conv2DTranspose, Lambda, Concatenate
import tensorflow.keras.backend as K

from utils import utils, load_data, evaluate

def get_second_stage(input_dim, gamma=5e-3, intermediate_dim=1536):
    latent_dim = input_dim

    # Loss
    def second_stage_loss(u_mean, u_log_var):
        def loss(x_true, x_pred):

            L_rec = utils.cos_sim(x_true, x_pred)
            L_KL = utils.KL(u_mean, u_log_var)(x_true, x_pred)

            return L_rec + gamma * L_KL
        return loss

    # Encoder
    z = Input(shape=(input_dim, ))

    h = Dense(intermediate_dim, activation='relu')(z)
    h = Dense(intermediate_dim, activation='relu')(h)

    h = Concatenate()([h, z])

    u_mean = Dense(latent_dim)(h)
    u_log_var = Dense(latent_dim)(h)

    u = Lambda(second_stage_sampling)([u_mean, u_log_var])
    second_encoder = Model(z, [u, u_mean, u_log_var])

    # Decoder
    u_in = Input(shape=(latent_dim, ))

    h = Dense(intermediate_dim, activation='relu')(u_in)
    h = Dense(intermediate_dim, activation='relu')(h)

    h = Concatenate()([h, u_in])

    z_decoded = Dense(input_dim)(h)
    second_decoder = Model(u_in, z_decoded)

    # VAE
    z_reconstructed = second_decoder(u)
    second_vae = Model(z, z_reconstructed)
    second_vae.add_loss(second_stage_loss(u_mean, u_log_var))

    return second_vae, second_encoder, second_decoder




