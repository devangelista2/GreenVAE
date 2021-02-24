import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Conv2D, Conv2DTranspose, Dense, \
    BatchNormalization, Reshape, Layer, ReLU, GlobalAveragePooling2D, Flatten, concatenate, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Models

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

def Encoder2(second_depth, second_dim, latent_dim, name='Encoder2'):
    with K.name_scope(name):
        enc2_input = Input(shape=(latent_dim,))
        t = enc2_input
        for i in range(second_depth):
            t = Dense(second_dim)(t)
            if not(i==second_depth-1):
                t = ReLU()(t)
        t = tf.concat([enc2_input, t], -1)

        mu_u = Dense(latent_dim)(t)
        logvar_u = Dense(latent_dim)(t)
        sd_u = tf.exp(.5 * logvar_u)
        u = mu_u + sd_u * K.random_normal(shape=K.shape(mu_u))
        encoder1 = Model(enc2_input, [mu_u,logvar_u,u])
    return(encoder1)

def Decoder2(second_depth, second_dim, latent_dim, name='Decoder2'):
    with K.name_scope(name):
        dec2_input = Input(shape=(latent_dim,))
        t = dec2_input
        for i in range(second_depth):
            t = Dense(second_dim,activation='relu')(t)
        t = tf.concat([dec2_input, t], -1)
        z_hat = Dense(latent_dim, name='z_hat')(t)
        decoder2 = Model(dec2_input,z_hat)
    return(decoder2)

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

def SecondStageModel(latent_dim,second_depth=3, second_dim=1024):
    encoder2 = Encoder2(second_depth=second_depth, second_dim=second_dim, latent_dim=latent_dim)
    decoder2 = Decoder2(second_depth=second_depth, second_dim=second_dim, latent_dim=latent_dim)

    z = Input(shape=(latent_dim,))
    gamma2 = Input(shape=())
    u_mean, u_log_var, u = encoder2(z)
    z_hat = decoder2(u)
    vae2 = Model([z,gamma2], z_hat)

    L_rec = 0.5 * K.sum(K.square(z - z_hat), axis=-1) / gamma2
    normalize_z = tf.nn.l2_normalize(z, 1)
    normalize_z_hat = tf.nn.l2_normalize(z_hat, 1)
    cos_similarity = -K.sum(normalize_z * normalize_z_hat, axis=-1)

    #bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    #L_bce = bce(normalize_z,normalize_z_hat)
    L_KL = 0.5 * K.sum(K.square(u_mean) + K.exp(u_log_var) - 1 - u_log_var, axis=-1)
    L_tot = K.mean(cos_similarity + .0075 * L_KL + .0005*L_rec)
    #L_tot = K.mean(L_bce + .004 * L_KL)  # + .01*L_rec)

    vae2.add_loss(L_tot)

    return (vae2, encoder2, decoder2)


####################################################################
# main
####################################################################

latent_dim = 64
vae1,encoder1,decoder1 = FirstStageModel((64,64,3),latent_dim,base_dim=64,num_scale=4)
#encoder1.summary()
#decoder1.summary()
vae1.summary()

def load_celeba():
    print("loading celeba")
    x_train = np.load(os.path.join(data_folder, 'celeba/train.npy')).astype(np.float32)
    x_test = np.load(os.path.join(data_folder, 'celeba/test.npy')).astype(np.float32)
    return (x_train / 255.,x_test / 255.)

x_train, x_test = load_celeba()

optimizer = Adam(learning_rate=.00001)
vae1.compile(optimizer=optimizer,loss=None,metrics=['mse'])

batch_size = 100
epochs = 10
load_w = True
weights_filename = 'two_stage_celeba_64_3SB_lat64_base64_light.hdf5'

if load_w:
    #estimate mse over 10000 images for the initial value of gamma
    vae1.load_weights('../demos/weights/'+weights_filename)
    xin = x_train[:10000]
    #the value of gamma is irrelevant for prediction
    xout = vae1.predict(xin)
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

        loss, bmseloss = vae1.train_on_batch([image_batch, gamma_in],image_batch)
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
    print("mse: ", mseloss)
    if True: #save weights after each epoch
        vae1.save_weights('weights/' + weights_filename)

#vae1.fit([x_train,gamma_in],x_train,epochs=1,callbacks=[GammaCallback()])
vae1.save_weights('weights/' + weights_filename)

#assert False

if False:
    gamma_in = np.ones((10, 1))
    mylist = [9482, 4753, 2381, 2752, 4634, 2840, 5460, 3485, 8684, 4421]
    #showlist = x_test[0:10]
    showlist = np.array([x_test[i] for i in mylist])
    plotting.show_reconstruction(vae1, x_test[0:10])
    #plotting.show_reconstruction(vae1,next(datagen.flow(x_train,None,batch_size=10)))
    plotting.show_generation(decoder1,latent_dim,save=True)

#encoder1.compile(optimizer=optimizer1)
z_mean, z_log_var, z = encoder1.predict(x_test)
mean_z_mean = np.mean(z_mean,axis=0)
z_var = np.exp(z_log_var)
print(z_var.shape)
inactives = evaluation.count_deactivated_variables(z_var)
var_law = evaluation.variance_law(z_mean,z_var)
print("inactive variables = ", inactives)
print("variance law = ", var_law)
actual_std = np.std(z_mean,axis=0)


if False:
    z_mean, z_log_var, z, factor = encoder1.predict(x_train[0:50000], batch_size=100)
    dist = np.zeros(10000)
    i = 24 #4421
    for j in range(10000):
        dist[j] = np.mean(np.square(z[i]-z[j]))
    print("mean={:.2f},min={:.2f},max={:.2f},std={:.2f}".format(np.mean(dist),np.min(dist),np.max(dist),np.std(dist)))
    edist = list(enumerate(dist))
    edist.sort(key=lambda x: x[1])
    print(edist[:10])
    seed = np.array([z[j] for (j, v) in edist[:10]])
    images = decoder1(seed)
    plotting.show_fig(images)
    print(edist[9990:10000])
    seed = np.array([z[j] for (j, v) in edist[9990:10000]])
    images = decoder1(seed)
    plotting.show_fig(images)

if False: #second stage
    print("computing latent space")
    z_mean, _, _ = encoder1.predict(x_train,batch_size=100)
    print("done")
    vae2, encoder2, decoder2 = SecondStageModel(latent_dim=64,second_depth=2,second_dim=1536)
    #encoder2.summary()
    #decoder2.summary()
    #vae2.summary()
    optimizer2 = Adam(learning_rate=.0001)
    vae2.compile(optimizer=optimizer2, loss=None, metrics=['mse','cosine_similarity'])
    batch_size2 = 100
    epochs2 = 30
    load_w2 = True
    weights_filename2 = 'two_stage_64_3SB_lat64_light_second_1536_celeba_bis.hdf5'
    #weights_filename2 = 'two_stage_48_3SB_lat64_second_1536_celeba.hdf5'
    #weights_filename2 = 'two_stage_64_4SB_lat100_second_1536.hdf5'  # second2 attorno a 91 (old arch)
    #weights_filename2 = 'two_stage_64_4SB_lat64_second_1536_celeba.hdf5' #second attorno a 86 con rescaling a 1
    #weights_filename2 = 'two_stage_64_4SB_lat64_second_1536.hdf5_bce'  # second attorno a 89 (old arch)

    if load_w2:
        vae2.load_weights('weights/' + weights_filename2)
        gamma2_in = np.ones(10000)
        zin = z_mean[:10000]
        zout = vae2.predict([zin,gamma2_in])
        mseloss = np.mean(np.square(zin - zout))
        vloss = evaluation.loss_variance(zin, zout)
        print("mse2 = {0:.4f}, vloss2 = {0:.4f}".format(mseloss, vloss))
    else:
        mseloss = 1.

    gamma2 = mseloss
    gamma2_in = np.ones(batch_size2)

    num_sample = np.shape(z_mean)[0]
    print('Num z = {}.'.format(num_sample))
    iteration_per_epoch2 = num_sample // batch_size2

    for epoch in range(epochs2):
        np.random.shuffle(z_mean)
        epoch_loss = 0
        for j in range(iteration_per_epoch2):
            z_batch = z_mean[j * batch_size2:(j + 1) * batch_size2]
            gamma2_in[:] = gamma2
            loss, bmseloss, cossim = vae2.train_on_batch([z_batch, gamma2_in], z_batch)
            epoch_loss += loss
            # we estimate mse as a weighted combination of the
            # the previous estimation and the minibatch mse
            # mseloss = min(mseloss,mseloss*.99+bmseloss*.01)
            mseloss = mseloss * .99 + bmseloss * .01
            gamma2 = min(mseloss, gamma2)
            if j % 50 == 0:
                # print("mse: ", mseloss)
                print(".", end='')
        epoch_loss /= iteration_per_epoch2
        print('Date: {date}\t'
              'Epoch: [Stage 1][{0}/{1}]\t'
              'Loss: {2:.4f}.'.format(epoch + 1, epochs2, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
        print("mse: {}, cos similarity: {}".format(mseloss,cossim))

    # vae1.fit([x_train,gamma_in],x_train,epochs=1,callbacks=[GammaCallback()])
    vae2.save_weights('weights/' + weights_filename2)

    z_mean, _, _ = encoder1.predict(x_test)
    u_mean, u_log_var, _ = encoder2.predict(z_mean)
    mean_u_mean = np.mean(u_mean, axis=0)
    u_var = np.exp(u_log_var)
    print(u_var.shape)
    inactives = evaluation.count_deactivated_variables(u_var)
    var_law = evaluation.variance_law(u_mean, u_var)
    print("inactive variables (2) = ", inactives)
    print("variance law (2) = ", var_law)

if False:
    eu = np.load("quadratic_train.npy")
    ma = np.load("mahalonois_train.npy")
    take = ma < 1.5*eu
    print("take no: {}".format(np.sum(take)))
    print(np.mean(ma))
    print(np.sum(ma*take)/np.sum(take))
    assert False
    takei = np.zeros((10000,32,32,3))
    j = 0
    for i in range(10000):
        while take[j] == False:
            j = j+1
        takei[i] = x_train[j]
        j = j+1
    images = vae1.predict(takei, batch_size=100)
    fidscore = evaluation.get_fid(x_test, images)
    print("selected = ", fidscore)

#evaluation.save_real_act_distribution(x_test[:10000])

if False:
    evaluation.save_real_act_distribution(x_test)
    print("mahalonobis")
    dist = np.zeros(50000)
    images = np.zeros((50000,32,32,3))
    for i in range(5):
        print(i)
        images[i*10000:(i+1)*10000] = vae1.predict(x_train[i*10000:(i+1)*10000], batch_size=100)
        dist[i*10000:(i+1)*10000] = evaluation.get_mahalonobis_distances(images[i*10000:(i+1)*10000])
    print("fine dist")
    np.save("quadratic_train.npy",dist)
    edist = list(enumerate(dist))
    edist.sort(key=lambda x: x[1])
    low = edist[0:10]
    print(low)
    images_orig = np.array([images[j] for (j, v) in low])
    plotting.show_fig(images_orig)
    high = edist[49990:50000]
    print(high)
    images_high = np.array([images[j] for (j, v) in high])
    plotting.show_fig(images_high)
    best = np.array([images[j] for (j, v) in edist[0:10000]])
    if True:
        fidscore = evaluation.get_fid(x_test,best)
        print("generated best = ", fidscore)

if False: #mahalonobis
    #evaluation.save_real_act_distribution(x_test)
    print("mahalonobis")
    #images = x_train[0:10000]
    #images = vae1.predict(x_train[0:10000], batch_size=100)
    seed = np.random.normal(0, 1, (10000, latent_dim))
    images = decoder1.predict(seed, batch_size=100)
    dist = evaluation.get_mahalonobis_distances(images)
    edist = list(enumerate(dist))
    edist.sort(key=lambda x: x[1])
    low = edist[5000:5010]
    print(low)
    images_orig = np.array([images[j] for (j, v) in low])
    plotting.show_fig(images_orig)
    high = edist[9990:10000]
    print(high)
    images_high = np.array([images[j] for (j, v) in high])
    plotting.show_fig(images_high)

if True: #plotting
    if True: #define vae2 and load weights
        vae2, encoder2, decoder2 = SecondStageModel(latent_dim=64,second_depth=2,second_dim=1536)
        weights_filename2 = 'two_stage_64_3SB_lat64_light_second_1536_celeba_bis.hdf5'
        vae2.load_weights('weights/' + weights_filename2)
    seed = np.random.normal(0, 1, (100, latent_dim))
    seed2 = decoder2.predict(seed, batch_size=100)
    if True: #normalize to 1
        std2 = np.std(seed2, axis=1)
        print(np.mean(std2))
        seed2 = seed2/np.mean(std2) #normalized to 1
    generated = decoder1.predict(seed2, batch_size=100)
    plotting.show_fig_square(generated,save_as="two_stage_plot2")

if False:
    true_images = x_test[0:10000]
    #generated = next(datagen.flow(x_train,None,batch_size=10000))
    #fidscore = evaluation.get_fid(true_images, generated * 2 - 1)
    #print("generated 1 = ", fidscore)
    if False:
        z_mean, z_log_var, z = encoder1.predict(x_train[0:10000],batch_size=100)
        if False: #normalize to distance 1
            sq =np.sqrt(np.mean(np.square(z_mean),axis=1))
            print(np.min(sq), np.max(sq))
            z_mean = z_mean/(sq[:,np.newaxis])
            sq = np.sqrt(np.mean(np.square(z_mean), axis=1))
            print(np.min(sq),np.max(sq))
        generated = decoder1.predict(z_mean,batch_size=100)
        fidscore = evaluation.get_fid(true_images, generated)
        print("reconstructed = ", fidscore)
    if False:
        seed = np.random.normal(0,1,(10000,latent_dim))
        generated = decoder1.predict(seed,batch_size=100)
        fidscore = evaluation.get_fid(true_images,generated)
        print("generated 1 = ", fidscore)
    if False:
        seed = np.random.normal(0, 1, (10000, latent_dim))
        seed2 = seed * actual_std
        print(actual_std.shape)
        print(seed2.shape)
        generated = decoder1.predict(seed2)
        fidscore = evaluation.get_fid(true_images, generated)
        print("generated 1 with std = ", fidscore)
    if True:
        seed = np.random.normal(0, 1, (10000, latent_dim))
        seed2 = decoder2.predict(seed, batch_size=100)
        if True: #normalize to 1
            std2 = np.std(seed2, axis=1)
            print(np.mean(std2))
            seed2 = seed2/np.mean(std2) #normalized to 1
        generated = decoder1.predict(seed2, batch_size=100)
        fidscore = evaluation.get_fid(true_images, generated)
        print("generated 2 = ", fidscore)
    if True:
        z_density = GaussianMixture(n_components=10, max_iter=200)
        print(z_mean.shape)
        print("Fitting GMM")
        z_density.fit(z_mean)
        seed2 = z_density.sample(10000)
        generated = decoder1.predict(seed2, batch_size=100)
        fidscore = evaluation.get_fid(true_images, generated)
        print("generated GMM = ", fidscore)
        


