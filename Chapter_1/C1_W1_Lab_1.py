import tensorflow as tf
from keras.utils.vis_utils import plot_model
import pydotplus as pydot
import graphviz
from keras.models import Model, Sequential
from keras import layers, Input


def build_model_with_sequential():
    # instantiate a Sequential class and linearly stack the Layers of your model
    seq_model = Sequential([layers.Flatten(input_shape=(28, 28)),
                            layers.Dense(128, activation=tf.nn.relu),
                            layers.Dense(10, activation=tf.nn.softmax)])

    return seq_model


def build_model_with_functional():
    # instantiate the input Tensor
    input_layer = Input(shape=(28, 28))

    # stack the Layers using syntax: new_Layer()(previous_Layer)
    flatten_layer = layers.Flatten()(input_layer)
    first_dense = layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
    output_layer = layers.Dense(10, activation=tf.nn.softmax)(first_dense)

    # declare inputs snd outputs
    func_model = Model(inputs=input_layer, outputs=output_layer)

    return func_model


model = build_model_with_sequential()

plot_model(model, show_shapes=True, show_layer_names=True, to_file='../models/model.png')

# prepare fashion mnist dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

# configure, train, and evaluate the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
