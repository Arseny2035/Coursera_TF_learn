# import keras.optimizers
import tensorflow as tf
# from keras import Model, layers, Input
from keras.layers import Lambda, Input, Flatten, Dense, Dropout
from keras.models import Model, model_from_json

# from keras.optimizers
# from keras.optimizers import RMSprop
from keras.datasets import fashion_mnist
from keras.utils.vis_utils import plot_model

from keras import backend as K
from keras.losses import Loss

import os

import pydot
import graphviz

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random


def create_pairs(x, digit_indices):
    # Positive and negative pairs creation.
    # Alternatives between positive and negative pairs.

    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


def create_pairs_os_set(images, labels):
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    # print('digit_indices: ', digit_indices)
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')

    return pairs, y


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def initialize_base_network():
    input = Input(shape=(28, 28,))
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=input, outputs=x)


def euclidian_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# Train the model
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)

    return contrastive_loss


class ContrastiveLoss(Loss):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Prepare train and test
tran_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Normalize values
tran_images = tran_images / 255.0
test_images = test_images / 255.0

# Create pairs on train and test sets
tr_pairs, tr_y = create_pairs_os_set(train_images, train_labels)
ts_pairs, ts_y = create_pairs_os_set(test_images, test_labels)

# To see simple pair
#######################
# this_pair = 8 # number - for example
#
# show_image(ts_pairs[this_pair][0])
# show_image(ts_pairs[this_pair][1])
#######################


# Print other pairs
show_image(tr_pairs[:, 0][0])
show_image(tr_pairs[:, 0][1])
show_image(ts_pairs[:, 0][0])
show_image(ts_pairs[:, 0][1])
#######################

# Set options for model save/load.
MUST_SAVE_THE_MODEL = True
LOAD_MODEL = True

OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate=0.001)
LOSS = ContrastiveLoss(margin=1)
METRICS = ['accuracy']

# Load or create model.
if LOAD_MODEL and os.path.exists('Data/Siamese/models/model.json') \
        and os.path.exists('Data/Siamese/models/model_weights.h5'):

    # Load saved model if it exists.
    json_file = open('Data/Siamese/models/model.json')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('Data/Siamese/models/model_weights.h5')
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    print("Model was load from disk!")

else:

    # If model was not created yet.
    base_network = initialize_base_network()
    plot_model(base_network, show_shapes=True, show_layer_names=True, to_file='Data/Siamese/models/base-model.png')

    # Create the left input and point to the network
    input_a = Input(shape=(28, 28,), name="left_input")
    vect_output_a = base_network(input_a)

    # Create the right input and point to the network
    input_b = Input(shape=(28, 28,), name="right_input")
    vect_output_b = base_network(input_b)

    # Measure the similarity of the two vector outputs
    output = Lambda(euclidian_distance, name='output_layer',
                    output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

    # Specify the inputs and output of the model
    model = Model([input_a, input_b], output)

    # Plot model graph
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='Data/Siamese/models/outer-model.png')

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, epochs=20, batch_size=128,
                        validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y))

    # Save new model if we had to.
    if MUST_SAVE_THE_MODEL:
        model_json = model.to_json()
        with open('Data/Siamese/models/model.json', 'w') as json_file:
            json_file.write(model_json)

        model.save_weights('Data/Siamese/models/model_weights.h5')
        print("Model saved to disk!")


# Compute classification accuracy with a fixed threshold on distance
def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

loss = model.evaluate(x=[ts_pairs[:, 0], ts_pairs[:, 1]], y=ts_y)

y_pred_train = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
train_accuracy = compute_accuracy(tr_y, y_pred_train)

y_pred_test = model.predict([ts_pairs[:, 0], ts_pairs[:, 1]])
test_accuracy = compute_accuracy(ts_y, y_pred_test)

print("Loss = {}, Train Accuracy = {}, Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))


def plot_metrics(metric_name, title, ylim=5.0):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)


plot_metrics(metric_name='loss', title="Loss", ylim=0.2)


# Matplotlib config
def visualize_images():
    plt.rc('image', cmap='gray_r')
    plt.rc('grid', linewidth=0)
    plt.rc('xtick', top=False, bottom=False, labelsize='large')
    plt.rc('ytick', top=False, bottom=False, labelsize='large')
    plt.rc('axes', facecolor='F8F8F8', titlesize='large', edgecolor='white')
    plt.rc('text', color='a8151a')
    plt.rc('figure', facecolor='F0F0F0')


def display_images(left, right, predictions, labels, title, n):
    plt.figure(figsize=(17, 3))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
    plt.grid(None)
    left = np.reshape(left, [n, 28, 28])
    left = np.swapaxes(left, 0, 1)
    left = np.reshape(left, [28, 28 * n])
    plt.imshow(left)
    plt.figure(figsize=(17, 3))
    plt.yticks([])
    plt.xticks([28 * x + 14 for x in range(n)], predictions)
    for i, t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if predictions[i] > 0.5: t.set_color('red')  # bad predictions in red color
    plt.grid(None)
    right = np.reshape(right, [n, 28, 28])
    right = np.swapaxes(right, 0, 1)
    right = np.reshape(right, [28, 28 * n])
    plt.imshow(right)


y_pred_train = np.square(y_pred_train)
n = 10
indexes = np.random.choice(len(y_pred_train), size=n)
display_images(tr_pairs[:, 0][indexes], tr_pairs[:, 1][indexes], y_pred_train[indexes], tr_y[indexes],
               "clothes and their dissimilarity", n)

##
