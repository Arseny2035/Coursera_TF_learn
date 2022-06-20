import tensorflow as tf
import keras.backend as K
import os
from keras.models import Model, model_from_json
from keras.utils.vis_utils import plot_model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def my_relu(x):
    return K.maximum(0.1, x)

def build_model_with_functional():
    funct_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(my_relu),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return funct_model



#################################################################################
# Set options for model save/load.
MUST_SAVE_THE_MODEL = True
MUST_LOAD_MODEL = True

MODEL_DIRECTORY = 'Data\Lambda_layer_MNIST\models'
MODEL_FILE = os.path.join(MODEL_DIRECTORY, 'saved_model.pb')
WEIGHTS_FILE = os.path.join(MODEL_DIRECTORY, 'model_weights.h5')

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

EPOCHS = 5
# BATCH_SIZE = 128

# Load or create model.
if MUST_LOAD_MODEL and os.path.exists(MODEL_FILE):

    json_file = open(MODEL_FILE, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)


    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(MODEL_DIRECTORY, 'loaded-model.png'))

    model.load_weights(WEIGHTS_FILE)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    print("Model was loaded from disk!")


    print(model.summary())

    model.evaluate(x_test, y_test)

else:

    model = build_model_with_functional()

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    print("Model was created!")

    try:
        os.makedirs(MODEL_DIRECTORY)
    except:
        pass
    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(MODEL_DIRECTORY, 'model.png'))
    print(model.summary())

    history = model.fit(x_train, y_train, epochs=EPOCHS)

    # Save new model if we had to.
    if MUST_SAVE_THE_MODEL:
        model_json = model.to_json()
        with open(MODEL_FILE, 'w') as json_file:
            json_file.write(model_json)
            json_file.close()
        model.save_weights(WEIGHTS_FILE)
        print("Model saved to disk!")