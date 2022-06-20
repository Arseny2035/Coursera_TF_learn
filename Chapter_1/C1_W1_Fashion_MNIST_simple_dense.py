import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras import layers, Input
import os
from keras.models import model_from_json


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


# prepare fashion mnist dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

print(mnist)

# Set options for model save/load.
MUST_SAVE_THE_MODEL = True
MUST_LOAD_MODEL = True

MODEL_DIRECTORY = 'Data\Fashion_MNIST_simple_dense\Models'
MODEL_FILE = os.path.join(MODEL_DIRECTORY, 'saved_model.pb')
WEIGHTS_FILE = os.path.join(MODEL_DIRECTORY, 'model_weights.h5')

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']
EPOCHS = 5

# Load or create model.
if MUST_LOAD_MODEL and os.path.exists(MODEL_FILE):

    json_file = open(MODEL_FILE, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(WEIGHTS_FILE)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    print("Model was loaded from disk!")

    print(model.summary())

    model.evaluate(test_images, test_labels)

else:

    model = build_model_with_sequential()

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    print("Model was created!")

    try:
        os.makedirs(MODEL_DIRECTORY)
    except:
        pass
    plot_model(model, show_shapes=True, show_layer_names=True, to_file=os.path.join(MODEL_DIRECTORY, 'model.png'))
    print(model.summary())

    history = model.fit(training_images, training_labels, epochs=EPOCHS)
    model.evaluate(test_images, test_labels)

    # Save new model if we had to.
    if MUST_SAVE_THE_MODEL:
        model_json = model.to_json()
        with open(MODEL_FILE, 'w') as json_file:
            json_file.write(model_json)
            json_file.close()
        model.save_weights(WEIGHTS_FILE)
        print("Model saved to disk!")
