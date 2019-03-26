import numpy as np
# from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
from sklearn.cluster import KMeans

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from main import data_loader

print('load ped data...')
X_train, Y_train, X_test, frame_map = data_loader.load_ped()
print(X_train.shape)
print(X_train[:5])
print('----------------------------------------------')
X_train = X_train[:, :, :, 0]
print(X_train.shape)
print(X_train[:5])
print('----------------------------------------------')
X_test = X_test[:, :, :, 0]
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)
print(X_train[:5])
print('----------------------------------------------')
X_test = np.expand_dims(X_test, axis=3)

# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

print('OK')
