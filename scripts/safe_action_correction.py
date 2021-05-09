#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger

# 构建并训练奖励预测模型
# from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

# 准备数据，用于测试重载的模型的准确性
# 加载数据集
data = np.load("trajectory.npy",allow_pickle=True)

print(np.shape(data))
print(np.shape(data[0][0][0]),np.shape(data[0][0][1]),np.shape(data[0][0][2]),np.shape(data[0][0][3]),np.shape(data[0][0][4]))


# 数据集预处理
data_vae_ = []
for tra in data:
  for t in tra:
    t_ = np.concatenate((t[0],t[1],[t[2]],[t[3]]), axis=-1)
    data_vae_.append(t_)

data_vae_ = np.array(data_vae_)
data_vae_.astype('float32')
data_vae_[np.isnan(data_vae_)] = 0
print(np.shape(data_vae_))

# 损失平滑化，便于后续训练
nums = np.shape(data_vae_)[0]
for i in range(nums):
  if data_vae_[i,-1] == 1:
    for j in range(1,100):
      if i-j>0:
        data_vae_[i-j,-1] = 1 - j/100


# VAE参数设置
batch_size = 64
# original_dim = data_vae.shape[1]
original_dim = 64
latent_dim = 20 # 隐变量取2维只是为了方便后面画图
intermediate_dim = 32
epochs = 50

# VAE模型定义
# Encoder
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)
# z_log_var = Dense(latent_dim)(h)
#
# # 重参数技巧-但是这里我们不需要随机性，因此只返回mean
# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = K.random_normal(shape=K.shape(z_mean))
#     # return z_mean + K.exp(z_log_var / 2) * epsilon
#     return z_mean
#
# z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#
# # 解码层，也就是生成器部分
# decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_mean = Dense(original_dim, activation=None)
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)

# vae = Model(x, x_decoded_mean)
encoder = Model(x, z_mean)
encoder.load_weights("/Users/liushaofan/PycharmProjects/safety-starter-agents/safe-model-para/encoder.h5")
# print(data_vae_[0])
# print(encoder.predict(data_vae_)[0])
# time.sleep(10000)

# 构建并训练奖励预测模型
model_r = Sequential()
# model.add(Embedding(max_features, output_dim=256))
model_r.add(LSTM(128))
model_r.add(Dropout(0.3))
# model.add(LSTM(64))
# model.add(Dropout(0.3))
model_r.add(Dense(1, activation=None))
model_r.load_weights("/Users/liushaofan/PycharmProjects/safety-starter-agents/safe-model-para/model_r.h5")

# model_r.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae'])
# model_r.fit(x_train, y_train_r, batch_size=256, epochs=3)

# 构建并训练风险预测模型
model_c = Sequential()
model_c.add(LSTM(128,return_sequences=True, return_state=False))
model_c.add(Dropout(0.3))
model_c.add(LSTM(64))
model_c.add(Dropout(0.3))
model_c.add(Dense(2, activation=K.softmax))
model_c.load_weights("/Users/liushaofan/PycharmProjects/safety-starter-agents/safe-model-para/model_c.h5")

# model_c.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae'])
# model_c.fit(x_train, y_train_c, batch_size=256, epochs=10)

# 构建并训练奖励预测模型
model_r2 = Sequential()
# model.add(Embedding(max_features, output_dim=256))
model_r2.add(LSTM(32))
model_r2.add(Dropout(0.2))
# model.add(LSTM(64))
# model.add(Dropout(0.3))
model_r2.add(Dense(1, activation=None))
model_r2.load_weights("/Users/liushaofan/PycharmProjects/safety-starter-agents/safe-model-para/model_r2.h5")

# model_r2.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae'])
# model_r2.fit(x_train2, y_train_r2, batch_size=128, epochs=5)

# 构建并训练风险预测模型
model_c2 = Sequential()
model_c2.add(LSTM(32,return_sequences=True, return_state=False))
model_c2.add(Dropout(0.3))
model_c2.add(LSTM(16))
model_c2.add(Dropout(0.2))
model_c2.add(Dense(2, activation=K.softmax))
model_c2.load_weights("/Users/liushaofan/PycharmProjects/safety-starter-agents/safe-model-para/model_c2.h5")

# model_c2.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae'])
# model_c2.fit(x_train2, y_train_c2, batch_size=64, epochs=10)

# 构建并训练风险预测模型
def safe_correction_metric(y_true, y_pred):
    pass

def safe_correction_loss(y_true, y_pred):
    pass

model_s = Sequential()
model_s.add(LSTM(32,return_sequences=True, return_state=False))
model_s.add(Dropout(0.3))
model_s.add(LSTM(16))
model_s.add(Dropout(0.2))
model_s.add(Dense(2, activation=K.tanh))
model_c2.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae'])

# x = Input(shape=(batch_size, , ))  # ??
h = Dense(intermediate_dim, activation='relu')(x)

z_mean = Dense(latent_dim)(h)

encoder = Model(x, z_mean)


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0

    seq_buffer = []
    beta = 0.

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)


        if len(seq_buffer)>100:
            cp = model_c.predict(seq_buffer[-100:])
            rp = model_r.predict(seq_buffer[-100:])
        cp2 = model_c2.predict(seq_buffer[-10:])
        rp2 = model_r2.predict(seq_buffer[-10:])

        model_s.fit(a, )

        if cp>beta:
            sa = model_s.predict()

        o_, r, d, info = env.step(a)

        t_ = [...]
        t = encoder.predict(t_)
        seq_buffer.append([t])

        o = o_
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default="/Users/liushaofan/PycharmProjects/safety-starter-agents/data/baseline/point-goal1-ppo")
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
