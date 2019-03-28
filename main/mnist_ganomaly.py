import matplotlib

matplotlib.use('Agg')

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def find_num(num):
    for index, label in enumerate(Y_test):
        if label == num:
            return index


"""# Data Transformation"""

(X1, Y1), (X2, Y2) = mnist.load_data()
X = np.vstack([X1, X2])
Y = np.hstack([Y1, Y2])

roc_auc_scores = []
prauc_scores = []

for num_remove in range(10):

    X_non_rem = []
    Y_non_rem = []
    X_rem = []
    Y_rem = []
    for i, label in enumerate(Y):
        if label != num_remove:
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

    """# Model"""

    latent_dim = 100
    input_shape = (28, 28, 1)


    def make_encoder():
        modelE = Sequential()
        modelE.add(Conv2D(32, kernel_size=(3, 2), padding="same", input_shape=input_shape))
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
        modelE.add(Dense(latent_dim))

        return modelE

    # Encoder 1

    enc_model_1 = make_encoder()
    img = Input(shape=input_shape)
    z = enc_model_1(img)
    encoder1 = Model(img, z)

    # Generator

    modelG = Sequential()
    modelG.add(Dense(128 * 7 * 7, input_dim=latent_dim))
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

    z = Input(shape=(latent_dim,))
    gen_img = modelG(z)
    generator = Model(z, gen_img)

    # Encoder 2

    enc_model_2 = make_encoder()
    img = Input(shape=input_shape)
    z = enc_model_2(img)
    encoder2 = Model(img, z)

    # Discriminator
    modelD = Sequential()
    modelD.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same"))
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

    discriminator = modelD

    optimizer = Adam(0.0001, 0.5)

    # Build and compile the discriminator
    discriminator.compile(loss=['binary_crossentropy'],
                          optimizer=optimizer,
                          metrics=['accuracy'])

    discriminator.trainable = False

    # First image encoding
    img = Input(shape=input_shape)
    z = encoder1(img)

    # Generate image from encoding
    img_ = generator(z)

    # Second image encoding
    z_ = encoder2(img_)

    # The discriminator takes generated images as input and determines if real or fake
    real = discriminator(img_)

    # Set up and compile the combined model
    # Trains generator to fool the discriminator
    # and decrease loss between (img, _img) and (z, z_)
    bigan_generator = Model(img, [real, img_, z_])
    bigan_generator.compile(loss=['binary_crossentropy', 'mean_absolute_error',
                                  'mean_squared_error'], optimizer=optimizer)

    batch_size = 128
    epochs = 40000

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_train_reshaped = np.reshape(X_train, (len(X_train) * 28, 28))
    # np.savetxt("X_train_ganomaly.txt", X_train_reshaped, fmt='%5s', delimiter=",")
    # np.savetxt("Y_train_ganomaly.txt", Y_train, fmt='%5s', delimiter=",")

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    # X_test_reshaped = np.reshape(X_test, (len(X_test) * 28, 28))
    # np.savetxt("X_test_ganomaly.txt", X_test_reshaped, fmt='%5s', delimiter=",")
    # np.savetxt("Y_test_ganomaly.txt", Y_test, fmt='%5s', delimiter=",")

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    # Adversarial ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    g_loss_list = []
    d_loss_list = []

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images and encode/decode/encode
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        z = encoder1.predict(imgs)
        imgs_ = generator.predict(z)

        # Train the discriminator (imgs are real, imgs_ are fake)
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(imgs_, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (z -> img is valid and img -> z is is invalid)
        g_loss = bigan_generator.train_on_batch(imgs, [real, imgs, z])

        # Plot the progress
        print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" %
              (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
        g_loss_list.append(g_loss)
        d_loss_list.append(d_loss)

    """# Finding Anomaly Scores"""

    plt.plot(np.asarray(g_loss_list)[:, 0], label='G loss')
    plt.plot(np.asarray(d_loss_list)[:, 0], label='D loss')
    plt.plot(np.asarray(d_loss_list)[:, 1], label='D accuracy')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig("loss/loss_%d.png" % num_remove, bbox_inches='tight', pad_inches=1)
    plt.close()

    # loss_all = np.asarray([np.asarray(g_loss_list)[:, 0],
    #                        np.asarray(d_loss_list)[:, 0], np.asarray(d_loss_list)[:, 1]])
    # np.savetxt("loss/loss.txt", loss_all, fmt='%5s', delimiter=",")

    z1_gen_ema = encoder1.predict(X_test)
    reconstruct_ema = generator.predict(z1_gen_ema)
    z2_gen_ema = encoder2.predict(reconstruct_ema)

    val_list = []
    for i in range(0, len(X_test)):
        val_list.append(np.mean(np.square(z1_gen_ema[i] - z2_gen_ema[i])))

    anomaly_labels = np.zeros(len(val_list))
    for i, label in enumerate(Y_test):
        if label == num_remove:
            anomaly_labels[i] = 1

    val_arr = np.asarray(val_list)
    val_probs = val_arr / max(val_arr)

    roc_auc = roc_auc_score(anomaly_labels, val_probs)
    prauc = average_precision_score(anomaly_labels, val_probs)
    roc_auc_scores.append(roc_auc)
    prauc_scores.append(prauc)

    print("ROC AUC SCORE FOR %d: %f" % (num_remove, roc_auc))
    print("PRAUC SCORE FOR %d: %f" % (num_remove, prauc))

    # plt.scatter(np.arange(10), anom_score_avgs)
    # plt.savefig("anom/anom_scores_%d.png" % num_remove)
    # plt.close()

    r, c = 2, 10

    input_list = []
    for i in range(0, 10):
        index = find_num(i)
        input_list.append(X_test[index])

    input_arr = np.asarray(input_list)
    z_gen_ema = encoder1.predict(input_arr)
    reconstruct_ema = generator.predict(z_gen_ema)

    fig, axs = plt.subplots(r, c)
    for j in range(c):
        input_pl = np.reshape(input_arr[j], (28, 28))
        axs[0, j].imshow(input_pl, cmap='gray')
        axs[0, j].axis('off')

        reconstruct_ema_pl = 0.5 * reconstruct_ema[j] + 0.5
        reconstruct_ema_pl = np.reshape(reconstruct_ema_pl, (28, 28))
        axs[1, j].imshow(reconstruct_ema_pl, cmap='gray')
        axs[1, j].axis('off')
    fig.savefig("mnist_imgs/recons_%d.png" % num_remove)
    plt.close()

    # Save the weights
    # generator.save_weights('models/%d_gen_weights_ganomaly.h5' % num_remove)
    # encoder1.save_weights('models/%d_enc1_weights_ganomaly.h5' % num_remove)
    # encoder2.save_weights('models/%d_enc2_weights_ganomaly.h5' % num_remove)
    # discriminator.save_weights('models/%d_dis_weights_ganomaly.h5' % num_remove)

    # Save the model architecture
    # with open('models/%d_gen_architecture_ganomaly.json' % num_remove, 'w') as f:
    #     f.write(generator.to_json())
    # with open('models/%d_enc1_architecture_ganomaly.json' % num_remove, 'w') as f:
    #     f.write(encoder1.to_json())
    # with open('models/%d_enc2_architecture_ganomaly.json' % num_remove, 'w') as f:
    #     f.write(encoder2.to_json())
    # with open('models/%d_dis_architecture_ganomaly.json' % num_remove, 'w') as f:
    #     f.write(discriminator.to_json())

    roc_auc_list = []
    for remove in range(0, 10):
        anomaly_labels = np.zeros(len(val_arr))
        for j, label in enumerate(Y_test):
            if label == remove:
                anomaly_labels[j] = 1
        roc_auc_list.append(roc_auc_score(anomaly_labels, val_probs))

    plt.figure(figsize=(10, 5))
    plt.scatter(['%s' % i for i in range(0, len(roc_auc_list))], roc_auc_list)
    plt.savefig('all_nums_auc/all_nums_auc_%d.png' % num_remove)
    plt.close()

print(roc_auc_scores)
plt.scatter(np.arange(len(roc_auc_scores)), roc_auc_scores)
plt.savefig("roc_auc_scores.png")
plt.close()

print(prauc_scores)
plt.scatter(np.arange(len(roc_auc_scores)), prauc_scores)
plt.savefig("prauc_scores.png")
plt.close()