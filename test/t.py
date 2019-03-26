import numpy as np
# from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
from sklearn.cluster import KMeans
import numpy as np

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
scaling_times = 2

x = [3, 5, 2, 8, 9, 11, 12, 10]
y = np.copy(x)
y.sort()
print(y)
print(x)
