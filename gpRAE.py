# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Dense, Flatten, Reshape, Conv2DTranspose, Lambda
import tensorflow.keras.backend as K

from utils import utils, load_data, evaluate
import two_stage

# Load data
data = "cifar10" # or celeba
if data == "cifar10":
    x_train, x_test = load_data.load_cifar10()
elif data == "celeba":
    x_train, x_test = load_data.load_celeba()
else:
    print('Dataset not found.')

# Parameters
input_dim = x_train[0].shape
latent_dim = 128

e_layers = 4
d_layers = 2
base_dim = 128

epochs = 150
batch_size = 100

gamma = 1e-3
lamb  = 1e-6


# Loss function
def rae_loss(x_true, x_pred, z):

    L_rec = utils.recon(x_true, x_pred)
    L_rae = K.mean(0.5 * K.sum(K.square(z), axis=-1))
    GP = K.sum(K.square(K.gradients(K.square(x_pred), z)), axis=-1)

    return L_rec + gamma * L_rae + lamb * GP

# Model Architecture
# ENCODER
x = Input(shape=input_dim)
h = x

for i in range(e_layers):

    h = Conv2D(base_dim * (2 ** i), 4, strides=(2, 2), padding='same')(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)

n_final_ch = K.int_shape(h)[-1]
h = Flatten()(h)

z = Dense(latent_dim)(h)

encoder = Model(x, [z, z_mean, z_log_var])

# DECODER
z_in = Input(shape=(latent_dim, ))

d = input_dim[0] // (2 ** (e_layers - d_layers))

h = Dense(d * d * n_final_ch)(z_in)
h = Reshape((d, d, n_final_ch))(h)
h = BatchNormalization()(h)
h = ReLU()(h)

for i in range(d_layers):

    h = Conv2DTranspose(n_final_ch * 2 ** (i+1), 4, strides=(2, 2), padding='same')(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)

x_decoded = Conv2DTranspose(input_dim[-1], 4, strides=(1, 1), padding='same', activation='sigmoid')(h)
decoder = Model(z_in, x_decoded)

# VAE
x_recon = decoder(z)
vae = Model(x, x_recon)
vae.add_loss(vae_loss(x, x_recon, z))

# Compile model
optimizer = utils.get_optimizer(x_train.shape[0] // batch_size)
vae.compile(optimizer=optimizer, loss=None, metrics=['mse'])

# Fit model
hist = vae.fit(x_train, None, batch_size=batch_size, epochs=epochs, verbose=1)
vae.save_weights('saved_weights/gpRAE_' + data +'.h5')


SECOND_STAGE = True
if SECOND_STAGE:
    z_train = encoder.predict(x_train)
    z_test  = encoder.predict(x_test)

    second_vae, second_encoder, second_decoder = two_stage.get_second_stage(latent_dim)

    # Compile model
    optimizer = utils.get_optimizer(z_train.shape[0] // batch_size, initial_lr=1e-3)
    second_vae.compile(optimizer=optimizer, loss=None, metrics=[utils.cos_sim])

    second_vae.fit(z_train, None, batch_size=batch_size, epochs=epochs)
    second_vae.save_weights('saved_weights/secondstage_gpVAE_' + data +'.h5')

GMM = False
if GMM:
    from sklearn.mixture import GaussianMixture

    z_density = GaussianMixture(n_components=10, max_iter=100)
    z_density.fit(z_train)
else:
    from scipy.stats import norm

    mean, std = norm.fit(z_train) # z_train is fitted to a gaussian N(mean, std)
    print("Learned Gaussian")

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
    elif GMM:
        z = utils.sample_from_GMM(z_density, 10000)
    else:
        z = np.random.normal(mean, std, (10000, latent_dim))
    x_gen = decoder.predict(z)

    gen_fid = evaluate.get_fid(x_test[:10000], x_gen)
    print('GEN FID:', gen_fid)

SHOW_REC_FID = False
if SHOW_REC_FID:
    x_recon = vae.predict(x_train[:10000])

    rec_fid = evaluate.get_fid(x_test[:10000], x_recon)
    print('REC FID:', rec_fid)



