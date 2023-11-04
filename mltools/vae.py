import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import argparse

class SimpleVAE:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        inputs = Input(shape=(self.input_dim,))
        x = Dense(128, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.0)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        z = Lambda(sampling)([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z])
        return encoder

    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(128, activation='relu')(latent_inputs)
        outputs = Dense(self.input_dim, activation='sigmoid')(x)
        decoder = Model(latent_inputs, outputs)
        return decoder

    def build_autoencoder(self):
        inputs = Input(shape=(self.input_dim,))
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        
        def vae_loss(inputs, outputs):
            reconstruction_loss = mse(inputs, outputs)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return reconstruction_loss + kl_loss
        
        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss=vae_loss)
        return autoencoder

    def train(self, data, epochs=100, batch_size=32):
        self.autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size)

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        generated_data = self.decoder.predict(noise)
        return generated_data


def main():
    parser = argparse.ArgumentParser(description='Simple Variational Autoencoder (VAE)')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension (default is 2)')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent dimension (default is 2)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default is 200)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default is 32)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of generated samples (default is 100)')
    args = parser.parse_args()

    data = np.random.randn(200, args.input_dim)  # Sample data
    vae = SimpleVAE(args.input_dim, args.latent_dim)
    vae.train(data, epochs=args.epochs, batch_size=args.batch_size)
    generated_samples = vae.generate_samples(args.num_samples)

if __name__ == "__main__":
    main()