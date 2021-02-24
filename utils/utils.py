import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add, Multiply
import numpy as np

def sampling(args):
    z_mean, z_log_var = args

    eps = K.random_normal(shape=K.shape(z_mean))
    return Add()([z_mean, Multiply()([K.exp(0.5 * z_log_var), eps])])

def recon(x_true, x_pred):
    input_dim = K.int_shape(x_true)[1:]

    x_true = K.reshape(x_true, (-1, np.prod(input_dim)))
    x_pred = K.reshape(x_pred, (-1, np.prod(input_dim)))

    return K.mean(0.5 * K.sum(K.square(x_true - x_pred), axis=-1))

def cos_sim(x_true, x_pred):
    import tensorflow as tf

    normalize_z = tf.nn.l2_normalize(x_true, 1)
    normalize_z_hat = tf.nn.l2_normalize(x_pred, 1)

    return K.mean(K.sum(normalize_z * normalize_z_hat, axis=-1))

def KL(z_mean, z_log_var):
    def kl(x_true, x_pred):
        return K.mean(0.5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=-1))
    return kl

def get_optimizer(steps_per_epoch, initial_lr=1e-4, halve_epochs=[80, 120, 150]):
	from tensorflow.keras.optimizers import Adam
	from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

	lr_schedule = PiecewiseConstantDecay([steps_per_epoch * epochs for epochs in halve_epochs], 
	                                     [initial_lr/(2**i) for i in range(len(halve_epochs)+1)])
	optimizer = Adam(learning_rate=lr_schedule)
	return optimizer

def sample_from_GMM(z_density, n):
    return z_density.sample(n)[0]









