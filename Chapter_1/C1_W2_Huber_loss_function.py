import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend as K

import os
from keras.models import Model, model_from_json, load_model
from keras.utils.vis_utils import plot_model

from keras.losses import Loss

import utils

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs=500, verbose=0)
print("optimizer='sgd', loss='mean_squared_error': ", model.predict([10.0]))


def my_huber_loss_with_threshold(threshold=1):
    def my_huber_loss(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)

    return my_huber_loss

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
print("optimizer='sgd', loss=my_huber_loss: ", model.predict([10.0]))



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

    def get_config(self):
        return {'threshold': self.threshold}
    #
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

# Set options for model save/load.
MUST_SAVE_THE_MODEL = True
MUST_LOAD_MODEL = True

# SAVE_LOAD_MODEL_METHOD = 'model.to_json'
SAVE_LOAD_MODEL_METHOD = 'model.save'

MODEL_DIRECTORY = 'Data\Huber_loss_function\Models'
MODEL_FILE = os.path.join(MODEL_DIRECTORY, 'saved_model.pb')
WEIGHTS_FILE = os.path.join(MODEL_DIRECTORY, 'model_weights.h5')

OPTIMIZER = 'sgd'
LOSS = MyHuberLoss(threshold=1)
# METRICS = ['accuracy']

EPOCHS = 500
# BATCH_SIZE = 32

# Load or create model.
if MUST_LOAD_MODEL and os.path.exists(MODEL_FILE):
    # Load saved model if it exists.

    if SAVE_LOAD_MODEL_METHOD == 'model.to_json':
        json_file = open(MODEL_FILE, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print("Model was load from disk!")

        model.load_weights(WEIGHTS_FILE)
        print("Weights loaded!")
        # model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    else:
        model = load_model(MODEL_FILE,
                           custom_objects={'MyHuberLoss': MyHuberLoss})
        print("Model was load from disk!")

    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(MODEL_DIRECTORY, 'loaded-model.png'))
    print(model.summary())
    print("optimizer='sgd', loss=MyHuberLoss(threshold=1): ", model.predict([10.0]))

else:
    # Creating model

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer=OPTIMIZER, loss=LOSS)
    print("Model was created!")

    try:
        os.makedirs(MODEL_DIRECTORY)
    except:
        pass
    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(MODEL_DIRECTORY, 'model.png'))
    print(model.summary())

    # TRAIN model
    history = model.fit(xs, ys, epochs=EPOCHS, verbose=0)
    print("optimizer='sgd', loss=MyHuberLoss(threshold=1): ", model.predict([10.0]))
    print('history: ', history)

    # SAVING model and weights (if necessary)
    if MUST_SAVE_THE_MODEL:
        if SAVE_LOAD_MODEL_METHOD == 'model.to_json':
            model_json = model.to_json()
            with open(MODEL_FILE, 'w') as json_file:
                json_file.write(model_json)
                print("Model saved!")
                json_file.close()
            model.save_weights(WEIGHTS_FILE)
            print("Weights saved!")
        else:
            model.save(MODEL_FILE)
            print("Model saved!")


def my_rmse(y_true, y_pred):
    error = y_true - y_pred
    sqr_error = K.square(error)
    mean_sqr_error = K.mean(sqr_error)
    sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
    return sqrt_mean_sqr_error