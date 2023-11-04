import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import argparse

class SimpleGAN:
    def __init__(self, input_dim, generator_output_dim, discriminator_output_dim):
        self.input_dim = input_dim
        self.generator_output_dim = generator_output_dim
        self.discriminator_output_dim = discriminator_output_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(self.generator_output_dim, activation='sigmoid'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.discriminator_output_dim, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = tf.keras.Input(shape=(self.input_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = tf.keras.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def train(self, data, epochs=100, batch_size=32):
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            generated_data = self.generator.predict(noise)
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            d_loss_real = self.discriminator.train_on_batch(data, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_labels)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
                self.plot_generated_data(epoch+1)

    def plot_generated_data(self, epoch):
        noise = np.random.normal(0, 1, (100, self.input_dim))
        generated_data = self.generator.predict(noise)
        plt.figure(figsize=(6, 6))
        plt.scatter(generated_data[:, 0], generated_data[:, 1])
        plt.title(f'Generated Data - Epoch {epoch}')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Simple GAN')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension (default is 2)')
    parser.add_argument('--generator_output_dim', type=int, default=2, help='Generator output dimension (default is 2)')
    parser.add_argument('--discriminator_output_dim', type=int, default=1, help='Discriminator output dimension (default is 1)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default is 200)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default is 32)')
    args = parser.parse_args()

    data = np.random.randn(200, args.input_dim)  # Sample real data
    gan = SimpleGAN(args.input_dim, args.generator_output_dim, args.discriminator_output_dim)
    gan.train(data, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()