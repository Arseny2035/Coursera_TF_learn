import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend as K

import utils

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=500, verbose=0)
print(model.predict([10.0]))


# def my_huber_loss_with_threshold(threshold=1):
#     def my_huber_loss(y_true, y_pred):
#         error = y_true - y_pred
#         is_small_error = tf.abs(error) <= threshold
#         small_error_loss = tf.square(error) / 2
#         big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
#         return tf.where(is_small_error, small_error_loss, big_error_loss)
#
#     return my_huber_loss

def my_huber_loss(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)



model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss=my_huber_loss)
model.fit(xs, ys, epochs=500, verbose=0)
print(model.predict([10.0]))

from tensorflow.keras.losses import Loss

class MyHuberLoss(Loss):
    threshold = 1

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss=MyHuberLoss(threshold=1))
model.fit(xs, ys, epochs=500, verbose=0)
print(model.predict([10.0]))


def my_rmse(y_true, y_pred):
    error = y_true - y_pred
    sqr_error = K.square(error)
    mean_sqr_error = K.mean(sqr_error)
    sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
    return sqrt_mean_sqr_error