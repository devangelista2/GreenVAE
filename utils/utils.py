import tensorflow.keras.backend as K

def sampling(args):
    z_mean, z_log_var = args

    b_size = K.shape(z_mean[0])
    latent_dim = K.shape(z_mean[1])
    eps = K.random_normal(shape=(b_size, latent_dim))

    return z_mean + K.exp(0.5 * z_log_var) * eps

def recon(x_true, x_pred):
    x_true = K.reshape(x_true, (-1, np.prod(input_dim)))
    x_pred = K.reshape(x_pred, (-1, np.prod(input_dim)))

    return K.mean(0.5 * K.sum(K.square(x_true - x_pred), axis=-1))

def KL(z_mean, z_log_var):
    def kl(x_true, x_pred):
        return K.mean(0.5 * K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=-1))
    return kl

def vae_loss(z_mean, z_log_var, gamma):
    def loss(x_true, x_pred):

        L_rec = recon(x_true, x_pred)
        L_KL = KL(z_mean, z_log_var)(x_true, x_pred)

        return L_rec + gamma * L_KL
    return loss