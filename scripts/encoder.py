from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 加载数据集
# (x_train, y_train_), (x_test, y_test_) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

data = np.load("trajectory.npy", allow_pickle=True)

print(np.shape(data))
print(np.shape(data[0][0][0]),np.shape(data[0][0][1]),np.shape(data[0][0][2]),np.shape(data[0][0][3]),np.shape(data[0][0][4]))

data_vae = []
for tra in data:
  for t in tra:
    t_ = np.concatenate((t[0],t[1],[t[2]],[t[3]]), axis=-1)
    data_vae.append(t_)

data_vae = np.array(data_vae)
data_vae.astype('float32')
np.random.shuffle(data_vae)

data_vae[np.isnan(data_vae)] = 0

data_size = data_vae.shape[0]
x_train = data_vae[:int(data_size*0.7)]
x_test = data_vae[int(data_size*0.7):]

print(np.shape(x_train))
print(np.shape(x_test))

batch_size = 64
original_dim = data_vae.shape[1]
latent_dim = 20 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 32
epochs = 50

# Encoder
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
# z_log_var = Dense(latent_dim)(h)

encoder = Model(x, z_mean)

encoder.load_weights("encoder_weights.h5")

# print(data_vae[0])
# latent_code = encoder.predict(data_vae, batch_size=64)
# print(latent_code[0])

encoded_data = encoder.predict(data_vae, batch_size=64)
print(np.shape(encoded_data))
encoded_data = np.reshape(encoded_data, newshape=[10,1000, 20])
print(np.shape(encoded_data))

data_rnn = []

for ind,tra in enumerate(encoded_data):
  l = len(tra) - 20
  for i in range(l-20-100):
    beg = i
    end = i+100
    pred = end+20-1
    x = tra[beg:end]
    y = np.concatenate(([data[ind][pred][2]],[data[ind][pred][3]]), axis=-1)  # [r,c]
    # break
    # print(x,y)
    data_rnn.append([x,y])

print(np.shape(data_rnn))
print(np.shape(data_rnn[0][0]))
print(np.shape(data_rnn[0][1]))

np.save("data_rnn.npy", data_rnn)