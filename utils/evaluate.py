import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3), weights='imagenet')

def get_forward_time(model, x_test, batch=100, n_epochs=50):
    import time

    n_samples = 10000
    nb = n_samples // batch

    times = []
    for epoch in range(n_epochs):
        print('Processing ', epoch + 1, '-th epoch.')
        initial_time = time.time()
        for i in range(nb):
            model.predict(x_test[i * batch : (i+1) * batch])
        times.append(time.time() - initial_time)
    times = np.array(times)

    print('Forward time: ', np.mean(times), ' +- ', np.std(times))
    return times


def get_inception_activations(inps, batch_size=100):
    n_batches = inps.shape[0] // batch_size

    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * batch_size:(i + 1) * batch_size]
        inpr = tf.image.resize(inp, (299, 299))
        inpr = inpr*2 - 1 #resize images in the interval [-1.1]
        act[i * batch_size:(i + 1) * batch_size] = model.predict(inpr, steps=1)

        print('Processed ' + str((i + 1) * batch_size) + ' images.')
    return act


def get_fid(images1, images2):
    from scipy.linalg import sqrtm

    shape = np.shape(images1)[1]
    if shape == 32:
        dataset = "cifar10"
    else:
        assert (shape == 64)
        dataset = "celeba"
    print("Computing FID for {} images".format(dataset))

    # activation for true images is always the same: we just compute it once
    if os.path.exists(dataset + "_act_mu.npy"):
        mu1 = np.load(dataset + "_act_mu.npy")
        sigma1 = np.load(dataset + "_act_sigma.npy")
    else:
        act1 = get_inception_activations(images1)
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        np.save(dataset + "_act_mu.npy", mu1)
        np.save(dataset + "_act_sigma.npy", sigma1)
    print('Done stage 1 of 2')

    act2 = get_inception_activations(images2)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    print('Done stage 2 of 2')

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # compute sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def count_deactivated_variables(z_var, treshold=0.8):
    z_var = np.mean(z_var, axis=0)
    return np.sum(z_var > treshold)

def get_loss_variance(x_true, x_recon):
    x_true = np.reshape(x_true, (-1, np.prod(x_true.shape[1:])))
    x_recon = np.reshape(x_recon, (-1, np.prod(x_recon.shape[1:])))

    var_true = np.mean(np.var(x_true, axis=1), axis=0)
    var_recon = np.mean(np.var(x_recon, axis=1), axis=0)

    return np.abs(var_true - var_recon)

def get_var_law(z_mean, z_var):
    return np.mean(np.var(z_mean, axis=0) + np.mean(z_var, axis=0))

