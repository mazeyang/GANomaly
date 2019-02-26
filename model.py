import os

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import logging

import numpy as np


class GanomalyModel():
    def __init__(self,
                 input_height=28, input_width=28, output_height=28, output_width=28,
                 dataset_name=None, log_dir='log'):

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.dataset_name = dataset_name
        self.log_dir = log_dir

        # if self.is_training:
        #     logging.basicConfig(filename='ganomaly_loss.log', level=logging.INFO)

        # load different datasets
        if self.dataset_name == 'mnist':
            (X_train, y_train), (_, _) = mnist.load_data()
            # Make the data range between 0~1.
            X_train = X_train / 255
            specific_idx = np.where(y_train == self.attention_label)[0]
            self.data = X_train[specific_idx].reshape(-1, 28, 28, 1)
            self.c_dim = 1
        else:
            assert 'Error in loading dataset'

        self.build_model()

    def build_generator(self, input_shape):
        image = Input(shape=input_shape, name='input_image')
        # Encoder 1.
        x = Conv2D(filters=self.df_dim * 2, kernel_size=5, strides=2, padding='same', name='g_encoder_h0_conv')(image)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 4, kernel_size=5, strides=2, padding='same', name='g_encoder_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 8, kernel_size=5, strides=2, padding='same', name='g_encoder_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder.
        x = Conv2D(self.gf_dim * 1, kernel_size=5, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim * 1, kernel_size=5, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim * 2, kernel_size=3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.c_dim, kernel_size=5, activation='sigmoid', padding='same')(x)
        return Model(image, x, name='R')

    def build_discriminator(self, input_shape):
        image = Input(shape=input_shape, name='d_input')
        x = Conv2D(filters=self.df_dim, kernel_size=5, strides=2, padding='same', name='d_h0_conv')(image)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 2, kernel_size=5, strides=2, padding='same', name='d_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 4, kernel_size=5, strides=2, padding='same', name='d_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 8, kernel_size=5, strides=2, padding='same', name='d_h3_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

        return Model(image, x, name='D')

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims)

        # Model to train D to discrimate real images.
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

        # Construct generator/R network.
        self.generator = self.build_generator(image_dims)
        img = Input(shape=image_dims)

        reconstructed_img = self.generator(img)

        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img)

        # Model to train Generator/R to minimize reconstruction loss and trick D to see
        # generated images as real ones.
        self.adversarial_model = Model(img, [reconstructed_img, validity])
        self.adversarial_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                       loss_weights=[self.r_alpha, 1],
                                       optimizer=optimizer)

        print('\ndiscriminator:')
        self.discriminator.summary()

        print('\nadversarial_model:')
        self.adversarial_model.summary()

    def train(self, epochs, batch_size=128, sample_interval=500):
        pass


if __name__ == '__main__':
    model = GanomalyModel(dataset_name='mnist', input_height=28, input_width=28)
    model.train(epochs=5, batch_size=128, sample_interval=500)
