from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split

import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
from skimage import io, transform

import data_loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

MODEL_DIR = '../model/'


class Ganomaly:
    def __init__(self, latent_dim=100, input_shape=(28, 28, 1), batch_size=128, epochs=40000, scaling_times=2):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaling_times = scaling_times

    '''
    def get_mnist_data(self):
        print('get train and tst data...')
        (X1, Y1), (X2, Y2) = mnist.load_data()
        X = np.vstack([X1, X2])
        Y = np.hstack([Y1, Y2])

        X_non_rem = []
        Y_non_rem = []
        X_rem = []
        Y_rem = []
        for i, label in enumerate(Y):
            if label != self.normal_class:
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

        print('train samples: %d, test samples: %d.' % (len(self.X_train), len(self.X_test)))
        print('[OK]')
    '''

    def get_ped_data(self):
        print('load ped data...')
        X_train, Y_train, X_test, frame_map = data_loader.load_ped()
        
        X_train = X_train[:, :, :, 0]
        X_test = X_test[:, :, :, 0]
        X_train = np.expand_dims(X_train, axis=3)
        X_test = np.expand_dims(X_test, axis=3)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5

        self.X_train, self.Y_train, self.X_test, self.frame_map = X_train, Y_train, X_test, frame_map
        print('OK')

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
        self.real_or_fake = self.discriminator(self.img_)

        self.ganomaly_model = Model(self.img, [self.real_or_fake, self.img_, self.z_])
        self.ganomaly_model.compile(loss=['binary_crossentropy', 'mean_absolute_error', 'mean_squared_error'],
                                    optimizer=self.optimizer)

        self.g_loss_list = []
        self.d_loss_list = []

        print('[OK]')

    def train(self):
        self.get_ped_data()
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
            g_loss = self.ganomaly_model.train_on_batch(imgs, [real, imgs, z])

            # Plot the progress
            print("epoch: %d [D loss: %f, acc: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
            self.g_loss_list.append(g_loss)
            self.d_loss_list.append(d_loss)

        # save models
        self.encoder1.save(os.path.join(MODEL_DIR, ('encoder1_%d.h5' % self.epochs)))
        self.encoder2.save(os.path.join(MODEL_DIR, ('encoder2_%d.h5' % self.epochs)))
        self.generator.save(os.path.join(MODEL_DIR, ('generator_%d.h5' % self.epochs)))

        # save loss img
        plt.plot(np.asarray(self.g_loss_list)[:, 0], label='G loss')
        plt.plot(np.asarray(self.d_loss_list)[:, 0], label='D loss')
        plt.plot(np.asarray(self.d_loss_list)[:, 1], label='D accuracy')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig("../imgs/loss_%d.png" % self.epochs, bbox_inches='tight', pad_inches=1)
        plt.close()

        print('[OK]')

    def eval(self):
        self.get_ped_data()
        self.encoder1 = load_model(os.path.join(MODEL_DIR, 'encoder1_3700.h5'))
        self.encoder2 = load_model(os.path.join(MODEL_DIR, 'encoder2_3700.h5'))
        self.generator = load_model(os.path.join(MODEL_DIR, 'generator_3700.h5'))
        print('evaluate on test data...')
        print('generate z1...')
        z1_gen_ema = self.encoder1.predict(self.X_test)
        print('generate fake images...')
        reconstruct_ema = self.generator.predict(z1_gen_ema)
        print('generate z2...')
        z2_gen_ema = self.encoder2.predict(reconstruct_ema)

        val_list = []
        for i in range(0, len(self.X_test)):
            val_list.append(np.mean(np.square(z1_gen_ema[i] - z2_gen_ema[i])))
        val_arr = np.asarray(val_list)

        # show box score
        y = np.copy(val_arr)
        y.sort()
        plt.plot([i for i in range(len(y))], y)
        plt.savefig('../imgs/box_score.png')

        val_probs = self.scale(val_arr)
        scores1, scores0 = [], []
        for k, v in self.frame_map.items():
            box_list = v['box_index']
            label = v['label']
            frame_score = np.sum([val_probs[i] for i in box_list])
            if label == 1:
                scores1.append(frame_score)
            else:
                scores0.append(frame_score)
        scores1.sort()
        scores0.sort()
        index1 = [i for i in range(len(scores1))]
        index0 = [i for i in range(len(scores0))]
        plt.title('Result Analysis')
        example_num = 5000
        plt.plot(index1[:example_num], scores1[:example_num], color='green', label='anomaly samples')
        plt.plot(index0[:example_num], scores0[:example_num], color='red', label='normal samples')
        plt.legend()  # 显示图例

        plt.xlabel('frame cnt')
        plt.ylabel('score')
        plt.savefig('../imgs/score_%d.png' % self.epochs)

        acc_list = []
        for threshold in np.arange(0.05, 5, 0.02):
            cnt1 = np.sum([1 if x >= threshold else 0 for x in scores1])
            cnt0 = np.sum([1 if x < threshold else 0 for x in scores1])
            acc = (cnt0 + cnt1) / (len(scores0) + len(scores1))
            print('debug:', cnt0, cnt1, len(scores0), len(scores1), acc)
            acc_list.append(acc)
        acc_list.sort()
        print(acc_list)

    def scale(self, x_):
        lenx, part = len(x_), int(0.9 * len(x_))
        if lenx <= 1:
            return x_
        x_ = (x_ - np.min(x_)) / (np.max(x_) - np.min(x_))
        x_.sort()
        y_ = x_[min(lenx - 2, part):]
        z = y_.reshape(-1, 1)
        km = KMeans(n_clusters=2, random_state=1)
        km.fit(z)
        lst = km.labels_
        pos = 0
        for idx in range(len(lst) - 1):
            if lst[idx] != lst[idx + 1]:
                pos = idx
        threshold = (y_[pos] + y_[pos + 1]) / 2
        x_norm = [i / self.scaling_times if i <= threshold else 1 - (1 - i) / self.scaling_times for i in x_]
        anomaly_cnt = sum([1 if i > threshold else 0 for i in x_])
        print('anomaly rate(box level):%.6f%%.' % (anomaly_cnt / lenx * 100))
        return x_norm


if __name__ == '__main__':
    model = Ganomaly(batch_size=128, epochs=3000)
    # model.train()
    model.eval()
