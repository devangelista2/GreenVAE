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

gamma = 0.024


# Loss function
def vae_loss(x_true, x_pred, z_mean, z_log_var):
    
    L_rec = utils.recon(x_true, x_pred)
    L_KL = utils.KL(z_mean, z_log_var)(x_true, x_pred)

    return L_rec + gamma * L_KL
    return loss

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

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(utils.sampling)([z_mean, z_log_var])

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
vae.add_loss(vae_loss(x, x_recon, z_mean, z_log_var))

# Compile model
optimizer = utils.get_optimizer(x_train.shape[0] // batch_size)
vae.compile(optimizer=optimizer, loss=None, metrics=['mse', utils.KL(z_mean, z_log_var)])


# Fit model
hist = vae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=1)
vae.save_weights('saved_weights/vanillaVAE_' + data +'.h5')


########################################################################################################################
# SHOW THE RESULTS
########################################################################################################################

SHOW_METRICS = False
if SHOW_METRICS:
    _, z_mean, z_log_var = encoder.predict(x_test)
    z_var = np.exp(z_log_var)

    n_deact = evaluate.count_deactivated_variables(z_var)
    print('We have a total of ', latent_dim, ' latent variables. ', count_deactivated_variables(z_var), ' of them are deactivated')

    var_law = evaluate.get_var_law(z_mean, z_var)
    print('Variance law has a value of: ', var_law)

    x_recon = vae.predict(x_train)
    print('We lost ', evaluate.loss_variance(x_test, x_recon), 'Variance of the original data')

SHOW_GEN_FID = False
if SHOW_GEN_FID:
    z = np.random.normal(0, 1, (10000, latent_dim))
    x_gen = decoder.predict(z)

    gen_fid = evaluate.get_fid(x_test[:10000], x_gen)
    print('GEN FID:', gen_fid)

SHOW_REC_FID = False
if SHOW_REC_FID:
    x_recon = vae.predict(x_train[:10000])

    rec_fid = evaluate.get_fid(x_test[:10000], x_recon)
    print('REC FID:', rec_fid)

SECOND_STAGE = False
if SECOND_STAGE:
    z_train = encoder.predict(x_train)[0]
    z_test  = encoder.predict(x_test)[0]

    second_vae, second_encoder, second_decoder = two_stage.get_second_stage(latent_dim)

    # Compile model
    optimizer = utils.get_optimizer(z_train.shape[0] // batch_size, initial_lr=1e-3)
    second_vae.compile(optimizer=optimizer, loss=None, metrics=[utils.cos_sim])




"""

# Generation
n = 10 #figure with n x n digits
digit_size = 32
figure = np.zeros((digit_size * n, digit_size * n, 3))
# we will sample n points randomly sampled

"""
#We want to sample z from q(z) = E_p(u)[q(z|u)]
#p(u) = N(0, I)
"""

u_sample = np.random.normal(size=(n**2, latent_dim), scale=1)
z_sample = second_decoder.predict(u_sample)
std2 = np.std(z_sample, axis=1)
z_sample = z_sample * 0.8 / np.mean(std2)
for i in range(n):
    for j in range(n):
        x_decoded = decoder.predict(np.array([z_sample[i + n * j]]))
        figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size, :] = x_decoded

plt.style.use('default')
plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig('twostage_VAE_CIFAR10_fixedgamma_generation.png')
plt.show()

import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
#from keras.applications.inception_v3 import preprocess_input
#from skimage.transform import resize
#from tensorflow.keras.models import load_model
#import os
#from matplotlib import pyplot

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3), weights='imagenet')

def get_inception_activations(inps, batch_size=100):
    n_batches = inps.shape[0]//batch_size
    act = np.zeros([inps.shape[0], 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * batch_size:(i + 1) * batch_size]
        inpr = tf.image.resize(inp, (299, 299))
        act[i * batch_size:(i + 1) * batch_size] = model.predict(inpr,steps=1)
        
        print('Processed ' + str((i+1) * batch_size) + ' images.')
    return act

def get_fid(images1, images2, use_preprocessed_test=False):
    print(images1.shape)
    print(images2.shape)
    print(type(images1))
    # calculate activations
    if not use_preprocessed_test:
        act1 = get_inception_activations(images1,batch_size=100)
    else:
        import pickle
        with open('./drive/MyDrive/Weights/test_FID.pickle', 'rb') as test_fid:
            act1 = pickle.load(test_fid)
    #print(np.shape(act1))
    act2 = get_inception_activations(images2,batch_size=100)
    # compute mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # compute sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

u_sample = np.random.normal(0, 1, size=(x_test.shape[0], latent_dim))
z_sample = second_decoder.predict(u_sample)
z_sample = second_decoder.predict(u_sample)
std2 = np.std(z_sample, axis=1)
z_sample = z_sample * 0.8 / np.mean(std2)
x_gen = decoder.predict(z_sample)

fid = get_fid(2 * x_test - 1, 2 * x_gen - 1, use_preprocessed_test=True)
print('\n FID: %.3f' % fid)

# Learn latent space distribution
z_train = encoder.predict(x_train)[0]

prior_for_qz = "GMM" # Choose between GMM or Gaussian
if prior_for_qz == "GMM":
    from sklearn.mixture import GaussianMixture

    z_density = GaussianMixture(n_components=10, max_iter=100)
    z_density.fit(z_train)

    print("Learned GMM")
elif prior_for_qz == "Gaussian":
    from scipy.stats import norm

    mean, std = norm.fit(z_train) # z_train is fitted to a gaussian N(mean, std)
    print("Learned Gaussian")
else:
    print("Distribution not found")

# Generation
n = 10 #figure with n x n digits
digit_size = 32
figure = np.zeros((digit_size * n, digit_size * n, 3))
# we will sample n points randomly sampled

if prior_for_qz == "GMM":
    z_sample = z_density.sample(n**2)
    z_sample = z_sample[0]
elif prior_for_qz == "Gaussian":
    z_sample = np.random.normal(size=(n**2, latent_dim), loc=mean, scale=std)
else:
    print("Distribution not found")

for i in range(n):
    for j in range(n):
        x_decoded = decoder.predict(np.array([z_sample[i + n * j]]))
        figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = x_decoded

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig('L2-RAE_CIFAR10_generation.png')
plt.show()

import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
#from keras.applications.inception_v3 import preprocess_input
#from skimage.transform import resize
#from tensorflow.keras.models import load_model
#import os
#from matplotlib import pyplot

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3), weights='imagenet')

def get_inception_activations(inps, batch_size=100):
    n_batches = inps.shape[0]//batch_size
    act = np.zeros([inps.shape[0], 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * batch_size:(i + 1) * batch_size]
        inpr = tf.image.resize(inp, (299, 299))
        act[i * batch_size:(i + 1) * batch_size] = model.predict(inpr,steps=1)
        
        print('Processed ' + str((i+1) * batch_size) + ' images.')
    return act

def get_fid(images1, images2, use_preprocessed_test=False):
    print(images1.shape)
    print(images2.shape)
    print(type(images1))
    # calculate activations
    if not use_preprocessed_test:
        act1 = get_inception_activations(images1,batch_size=100)
    else:
        import pickle
        with open('./drive/MyDrive/Weights/test_FID.pickle', 'rb') as test_fid:
            act1 = pickle.load(test_fid)
    #print(np.shape(act1))
    act2 = get_inception_activations(images2,batch_size=100)
    # compute mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # compute sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

if prior_for_qz == "GMM":
    z_sample = z_density.sample(x_test.shape[0])
elif prior_for_qz == "Gaussian":
    z_sample = np.random.normal(size=(x_test.shape[0], latent_dim), loc=mean, scale=std)
else:
    print("Distribution not found")
x_gen = decoder.predict(z_sample)

fid = get_fid(2 * x_test - 1, 2 * x_gen - 1, use_preprocessed_test=True)
print('\n FID: %.3f' % fid)
#x_gen = vae.predict(x_train[:10000])
#rec_fid = get_fid(2 * x_test - 1, 2 * x_gen - 1, use_preprocessed_test=True)
#print('\n Train FID: %.3f' % rec_fid)
"""