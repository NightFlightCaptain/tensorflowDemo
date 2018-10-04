# -*- coding:utf-8 -*-
# __author__ = 'wanhaoran'

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import matplotlib.pyplot as plt

# 开启eager
tf.enable_eager_execution()

train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]

n_samples = len(train_X)

# Parameters
learning_rate = 0.01
display_step = 100
num_steps = 1000

W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())


def linear_regression(inputs):
    return inputs * W + b


def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
