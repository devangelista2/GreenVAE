import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, \
    BatchNormalization, Reshape, ReLU, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from utils import utils, load_data, evaluate
import two_stage
import numpy as np
import os

data_folder = '../GAN_FACES/data'

def ResBlock(out_dim, depth=2, kernel_size=3, name='ResBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(depth):
            y = BatchNormalization(momentum=.999,epsilon=1e-5)(y)
            y = ReLU()(y)
            y = Conv2D(out_dim,kernel_size,padding='same')(y)
        s = Conv2D(out_dim, kernel_size,padding='same')(inputs)
      return y + s
    return(body)

def ResFcBlock(out_dim, depth=2, name='ResFcBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(depth):
            y = BatchNormalization(momentum=.999,epsilon=1e-5)(y)
            y = ReLU()(y)
            y = Dense(out_dim)(y)
        s = Dense(out_dim)(inputs)
      return y + s
    return(body)

def ScaleBlock(out_dim, block_per_scale=1, depth_per_block=2, kernel_size=3, name='ScaleBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(block_per_scale):
            y = ResBlock(out_dim,depth=depth_per_block, kernel_size=kernel_size)(y)
      return y
    return (body)

def ScaleFcBlock(out_dim, block_per_scale=1, depth_per_block=2, name='ScaleFcBlock'):
    def body(inputs, **kwargs):
      with K.name_scope(name):
        y = inputs
        for i in range(block_per_scale):
            y = ResFcBlock(out_dim, depth=depth_per_block)(y)
      return y
    return(body)

def Encoder1(input_shape, base_dim, kernel_size, num_scale, block_per_scale, depth_per_block,
             fc_dim, latent_dim, name='Encoder1'):
    with K.name_scope(name):
        dim = base_dim
        enc_input = Input(shape=input_shape)
        y = Conv2D(dim,kernel_size,padding='same',strides=2)(enc_input)
        for i in range(num_scale-1):
            y = ScaleBlock(dim, block_per_scale, depth_per_block, kernel_size)(y)
            if i != num_scale - 1:
                dim *= 2
                y = Conv2D(dim,kernel_size,strides=2,padding='same')(y)

        y = GlobalAveragePooling2D()(y)
        y = ScaleFcBlock(fc_dim,1,depth_per_block)(y)

        mu_z = Dense(latent_dim)(y)
        logsd_z = Dense(latent_dim)(y)
        logvar_z = 2*logsd_z
        sd_z = tf.exp(logsd_z)
        z = mu_z + K.random_normal(shape=(K.shape(mu_z)[0],latent_dim)) * sd_z
        encoder = Model(enc_input,[mu_z,logvar_z,z])
    return encoder

def Decoder1(inp_shape, latent_dim, dims, scales, kernel_size, block_per_scale, depth_per_block, name='Decoder1'):
    base_wh = 4
    fc_dim = base_wh * base_wh * dims[0]
    data_depth = inp_shape[-1]

    with K.name_scope(name):
        dec_input = Input(shape=(latent_dim,))
        y = Dense(fc_dim)(dec_input)
        y = Reshape((base_wh,base_wh,dims[0]))(y)

        for i in range(len(scales) - 1):
            y = Conv2DTranspose(dims[i+1], kernel_size, strides=2, padding='same')(y)
            if not(i == len(scales)-2):
                y = ScaleBlock(dims[i+1],block_per_scale, depth_per_block, kernel_size)(y)

        x_hat = Conv2D(data_depth, kernel_size, 1, padding='same', activation='sigmoid')(y)
        decoder1 = Model(dec_input,x_hat)

    return(decoder1)

def FirstStageModel(input_shape,latent_dim,base_dim=32,fc_dim=512, kernel_size=3,num_scale=3,block_per_scale=1,depth_per_block=2):
    # base_dim refers to channels; they are doubled at each downscaling
    desired_scale = input_shape[1]
    scales, dims = [], []
    current_scale, current_dim = 4, base_dim
    while current_scale <= desired_scale:
        scales.append(current_scale)
        dims.append(current_dim)
        current_scale *= 2
        current_dim = min(current_dim * 2, 1024)
    assert (scales[-1] == desired_scale)
    dims = list(reversed(dims))
    print(dims,scales)

    encoder1 = Encoder1(input_shape, base_dim, kernel_size, num_scale, block_per_scale, depth_per_block,
                 fc_dim, latent_dim)
    decoder1 = Decoder1(input_shape, latent_dim, dims, scales, kernel_size, block_per_scale, depth_per_block)

    x = Input(shape=input_shape)
    gamma = Input(shape=()) #adaptive gamma parameter
    z_mean, z_log_var, z = encoder1(x)
    x_hat = decoder1(z)
    vae1 = Model([x,gamma],x_hat)

    #loss
    k = (2 * input_shape[1] / latent_dim) ** 2
    L_rec = 0.5 * K.sum(K.square(x-x_hat), axis=[1,2,3]) / gamma
    L_KL = 0.5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=-1)
    L_tot = K.mean(L_rec + k * L_KL)

    vae1.add_loss(L_tot)

    return(vae1,encoder1,decoder1)

####################################################################
# Train
####################################################################

latent_dim = 64
vae,encoder,decoder = FirstStageModel((64,64,3),latent_dim,base_dim=64,num_scale=4)
#encoder.summary()
#decoder.summary()
vae.summary()

def load_celeba():
    print("loading celeba")
    x_train = np.load(os.path.join(data_folder, 'celeba/train.npy')).astype(np.float32)
    x_test = np.load(os.path.join(data_folder, 'celeba/test.npy')).astype(np.float32)
    return (x_train / 255.,x_test / 255.)

x_train, x_test = load_celeba()

optimizer = Adam(learning_rate=.00001)
vae.compile(optimizer=optimizer,loss=None,metrics=['mse'])

batch_size = 100
epochs = 0
load_w = True
weights_filename = 'two_stage_celeba_64_3SB_lat64_base64_light.hdf5'

if load_w:
    #estimate mse over 10000 images for the initial value of gamma
    vae.load_weights('saved_weights/'+weights_filename)
    xin = x_train[:10000]
    #the value of gamma is irrelevant for prediction
    xout = vae.predict(xin)
    mseloss = np.mean(np.square(xin - xout))
    print("mse = {0:.4f}".format(mseloss))
else:
    mseloss = 1.

gamma = mseloss
gamma_in = np.zeros(batch_size)

num_sample = np.shape(x_train)[0]
print('Num Sample = {}.'.format(num_sample))
iteration_per_epoch = num_sample // batch_size

for epoch in range(epochs):
    np.random.shuffle(x_train)
    epoch_loss = 0
    for j in range(iteration_per_epoch):
        image_batch = x_train[j*batch_size:(j+1)*batch_size]
        gamma_in[:] = gamma

        loss, bmseloss = vae.train_on_batch([image_batch, gamma_in],image_batch)
        epoch_loss += loss
        #we estimate mse as a running average with the minibatch loss
        mseloss = mseloss * .99 + bmseloss * .01
        gamma = min(mseloss, gamma)
        if j % 50 == 0:
            #print("mse: ", mseloss)
            print(".",end='')
    epoch_loss /= iteration_per_epoch
    print('Date: {date}\t'
          'Epoch: [Stage 1][{0}/{1}]\t'
          'Loss: {2:.4f}.'.format(epoch+1, epochs, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
    print("MSE: ", mseloss)
    if True: #save weights after each epoch
        vae.save_weights('saved_weights/' + weights_filename)

vae.save_weights('saved_weights/' + weights_filename)

# Second stage and GMM
SECOND_STAGE = False
latent_dim = 64

if SECOND_STAGE:
    z_train = encoder.predict(x_train)[0]
    z_test  = encoder.predict(x_test)[0]

    second_vae, second_encoder, second_decoder = two_stage.get_second_stage(latent_dim)

    # Compile model
    optimizer = utils.get_optimizer(z_train.shape[0] // batch_size, initial_lr=1e-3)
    second_vae.compile(optimizer=optimizer, loss=None, metrics=[utils.cos_sim])

    weights_filename2 = 'two_stage_64_3SB_celeba_bis.hdf5'
    #second_vae.load_weights('saved_weights/' + weights_filename2)
    second_vae.fit(z_train, None, batch_size=batch_size, epochs=epochs)
    second_vae.save_weights('saved_weights/' + weights_filename2)

# GMM
GMM = False
if GMM:
    from sklearn.mixture import GaussianMixture

    z_density = GaussianMixture(n_components=10, max_iter=100)
    z_density.fit(z_train)

########################################################################################################################
# SHOW THE RESULTS
########################################################################################################################

SHOW_METRICS = True
if SHOW_METRICS:
    z_mean, z_log_var, _ = encoder.predict(x_test)
    z_var = np.exp(z_log_var)

    n_deact = evaluate.count_deactivated_variables(z_var)
    print('We have a total of ', latent_dim, ' latent variables. ', n_deact, ' of them are deactivated')

    var_law = evaluate.get_var_law(z_mean, z_var)
    print('Variance law has a value of: ', var_law)

    x_recon = vae.predict(x_test)
    print('We lost ', evaluate.get_loss_variance(x_test, x_recon), 'Variance of the original data')

SHOW_GEN_FID = False
if SHOW_GEN_FID:
    if SECOND_STAGE:
        u_sample = np.random.normal(0, 1, (10000, latent_dim))
        z = second_decoder.predict(u_sample)
    elif GMM:
        z = utils.sample_from_GMM(z_density, 10000)
    else:
        z = np.random.normal(0, 1, (10000, latent_dim))
    x_gen = decoder.predict(z)

    gen_fid = evaluate.get_fid(x_test[:10000], x_gen)
    print('GEN FID:', gen_fid)

SHOW_REC_FID = True
if SHOW_REC_FID:
    x_recon = vae.predict(x_train[:10000])

    rec_fid = evaluate.get_fid(x_test[:10000], x_recon)
    print('REC FID:', rec_fid)


