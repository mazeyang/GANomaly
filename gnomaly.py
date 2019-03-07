from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# Model
class Ganomaly:
    def __init__(self, latent_dim=100, input_shape=(28, 28, 1), batch_size=128, epochs=40, anomaly_class=2):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.anomaly_class = anomaly_class

    def get_data(self):
        print('get train and tst data...')
        (X1, Y1), (X2, Y2) = mnist.load_data()
        X = np.vstack([X1, X2])
        Y = np.hstack([Y1, Y2])

        X_non_rem = []
        Y_non_rem = []
        X_rem = []
        Y_rem = []
        for i, label in enumerate(Y):
            if label != self.anomaly_class:
                X_non_rem.append(X[i])
                Y_non_rem.append(Y[i])
            else:
                X_rem.append(X[i])
                Y_rem.append(Y[i])

        X_non_rem = np.asarray(X_non_rem)
        X_rem = np.asarray(X_rem)

        X_train, X_test_rem, Y_train, Y_test_rem = train_test_split(
            X_non_rem, Y_non_rem, test_size=0.2, random_state=42)

        X_test = np.vstack([X_test_rem, X_rem])
        Y_test = np.hstack([Y_test_rem, Y_rem])

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5

        # expand dimensions
        X_train = np.expand_dims(X_train, axis=3)
        X_test = np.expand_dims(X_test, axis=3)

        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test

        print('[OK]')

    def basic_encoder(self):
        modelE = Sequential()
        modelE.add(Conv2D(32, kernel_size=(3, 2), padding="same", input_shape=self.input_shape))
        modelE.add(BatchNormalization(momentum=0.8))
        modelE.add(Activation("relu"))
        modelE.add(MaxPooling2D(pool_size=(2, 2)))
        modelE.add(Conv2D(64, kernel_size=(3, 2), padding="same"))
        modelE.add(BatchNormalization(momentum=0.8))
        modelE.add(Activation("relu"))
        modelE.add(MaxPooling2D(pool_size=(2, 1)))
        modelE.add(Conv2D(128, kernel_size=(3, 2), padding="same"))
        modelE.add(BatchNormalization(momentum=0.8))
        modelE.add(Activation("relu"))
        modelE.add(Flatten())
        modelE.add(Dense(self.latent_dim))

        return modelE

    # Encoder 1
    def make_encoder_1(self):
        enc_model_1 = self.basic_encoder()
        img = Input(shape=self.input_shape)
        z = enc_model_1(img)
        encoder1 = Model(img, z)
        return encoder1

    # Generator
    def make_generator(self):
        modelG = Sequential()
        modelG.add(Dense(128 * 7 * 7, input_dim=self.latent_dim))
        modelG.add(BatchNormalization(momentum=0.8))
        modelG.add(LeakyReLU(alpha=0.2))
        modelG.add(Reshape((7, 7, 128)))
        modelG.add(Conv2DTranspose(128, kernel_size=(3, 2), strides=2, padding="same"))
        modelG.add(BatchNormalization(momentum=0.8))
        modelG.add(LeakyReLU(alpha=0.2))
        modelG.add(Conv2DTranspose(64, kernel_size=(3, 2), strides=2, padding="same"))
        modelG.add(BatchNormalization(momentum=0.8))
        modelG.add(LeakyReLU(alpha=0.2))
        modelG.add(Conv2DTranspose(1, kernel_size=(3, 2), strides=1, padding="same", activation='tanh'))

        z = Input(shape=(self.latent_dim,))
        gen_img = modelG(z)
        generator = Model(z, gen_img)
        return generator

    # Encoder 2
    def make_encoder_2(self):
        enc_model_2 = self.basic_encoder()
        img = Input(shape=self.input_shape)
        z = enc_model_2(img)
        encoder2 = Model(img, z)
        return encoder2

    # Discriminator
    def make_discriminator(self):
        modelD = Sequential()
        modelD.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.input_shape, padding="same"))
        modelD.add(LeakyReLU(alpha=0.2))
        modelD.add(Dropout(0.25))
        modelD.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        modelD.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        modelD.add(BatchNormalization(momentum=0.8))
        modelD.add(LeakyReLU(alpha=0.2))
        modelD.add(Dropout(0.25))
        modelD.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        modelD.add(BatchNormalization(momentum=0.8))
        modelD.add(LeakyReLU(alpha=0.2))
        modelD.add(Dropout(0.25))
        # modelD.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        # modelD.add(BatchNormalization(momentum=0.8))
        # modelD.add(LeakyReLU(alpha=0.2))
        # modelD.add(Dropout(0.25))
        modelD.add(Flatten())
        modelD.add(Dense(1, activation='sigmoid'))

        return modelD

    def make_components(self):
        print('make components...')

        self.optimizer = Adam(0.0001, 0.5)

        self.discriminator = self.make_discriminator()
        self.discriminator.trainable = True

        # Build and compile the discriminator
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=self.optimizer, metrics=['accuracy'])
        self.discriminator.trainable = False

        # First image encoding
        self.img = Input(shape=self.input_shape)

        self.encoder1 = self.make_encoder_1()
        self.z = self.encoder1(self.img)

        self.generator = self.make_generator()
        self.img_ = self.generator(self.z)

        self.encoder2 = self.make_encoder_2()
        self.z_ = self.encoder2(self.img_)

        # The discriminator takes generated images as input and determines if real or fake
        self.real = self.discriminator(self.img_)

        self.bigan_generator = Model(self.img, [self.real, self.img_, self.z_])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'mean_absolute_error', 'mean_squared_error'],
                                     optimizer=self.optimizer)

        self.g_loss_list = []
        self.d_loss_list = []

        print('[OK]')

    def train(self):
        self.get_data()
        self.make_components()

        # Adversarial ground truths
        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        print('start train...')
        for epoch in range(self.epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images and encode/decode/encode
            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
            imgs = self.X_train[idx]
            z = self.encoder1.predict(imgs)
            imgs_ = self.generator.predict(z)

            # Train the discriminator (imgs are real, imgs_ are fake)
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(imgs_, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch(imgs, [real, imgs, z])

            # Plot the progress
            print("epoch: %d [D loss: %f, acc: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
            self.g_loss_list.append(g_loss)
            self.d_loss_list.append(d_loss)

        print('[OK]')

    def show_loss(self):
        plt.plot(np.asarray(self.g_loss_list)[:, 0], label='G loss')
        plt.plot(np.asarray(self.d_loss_list)[:, 0], label='D loss')
        plt.plot(np.asarray(self.d_loss_list)[:, 1], label='D accuracy')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig("loss/loss_%d.png" % self.anomaly_class, bbox_inches='tight', pad_inches=1)
        plt.close()

    def find_scores(self):
        print('find_scores...')
        print('%d test samples.' % len(self.X_test))
        print('generate z1...')
        z1_gen_ema = self.encoder1.predict(self.X_test)
        print('generate fake images...')
        reconstruct_ema = self.generator.predict(z1_gen_ema)
        print('generate z2...')
        z2_gen_ema = self.encoder2.predict(reconstruct_ema)

        val_list = []
        for i in range(0, len(self.X_test)):
            val_list.append(np.mean(np.square(z1_gen_ema[i] - z2_gen_ema[i])))

        anomaly_labels = np.zeros(len(val_list))
        for i, label in enumerate(self.Y_test):
            if label == self.anomaly_class:
                anomaly_labels[i] = 1

        val_arr = np.asarray(val_list)
        val_probs = val_arr / max(val_arr)

        roc_auc = roc_auc_score(anomaly_labels, val_probs)
        prauc = average_precision_score(anomaly_labels, val_probs)
        # roc_auc_scores.append(roc_auc)
        # prauc_scores.append(prauc)

        print("ROC AUC SCORE FOR [%d](anomaly class): %f" % (self.anomaly_class, roc_auc))
        print("PRAUC SCORE FOR [%d](anomaly class): %f" % (self.anomaly_class, prauc))
        print('[OK]')


if __name__ == '__main__':
    model = Ganomaly(batch_size=128, epochs=2, anomaly_class=2)
    model.train()
    model.show_loss()
    model.find_scores()
