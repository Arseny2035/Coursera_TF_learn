import tensorflow as tf
import tensorflow_datasets as tfds

import os
from keras.models import Model, model_from_json, load_model
from keras.utils.vis_utils import plot_model

from keras.layers import Conv2D, Activation, BatchNormalization, Add, \
    MaxPool2D, GlobalAveragePooling2D, Dense


class IdentifyBlock(Model):
    def __init__(self, filters, kernel_size):
        super(IdentifyBlock, self).__init__(name='')

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)
        return x


class ResNet(Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, 7, padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.max_pool = MaxPool2D((3, 3))

        self.id1a = IdentifyBlock(64, 3)
        self.id1b = IdentifyBlock(64, 3)

        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)
        return self.classifier(x)


def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']


# Set options for model save/load.
MUST_SAVE_THE_MODEL = True
MUST_LOAD_MODEL = True

# SAVE_LOAD_MODEL_METHOD = 'model.to_json'
SAVE_LOAD_MODEL_METHOD = 'model.save'

MODEL_DIRECTORY = 'Data\Residual_network_MNIST\Models'
MODEL_FILE = os.path.join(MODEL_DIRECTORY, 'saved_model.pb')
WEIGHTS_FILE = os.path.join(MODEL_DIRECTORY, 'model_weights.h5')

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

EPOCHS = 1
BATCH_SIZE = 32

# Create DATASET
splits = ['train[:90%]', 'train[90%:]']
trainSet, testSet = tfds.load('mnist', split=splits)

trainSet = trainSet.map(preprocess).batch(BATCH_SIZE)
testSet = testSet.map(preprocess).batch(BATCH_SIZE)

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
        model = load_model(MODEL_FILE)
        print("Model was load from disk!")

    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(MODEL_DIRECTORY, 'loaded-model.png'))
    print(model.summary())

else:
    # Creating model

    model = ResNet(10)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    print("Model was created!")

    try:
        os.makedirs(MODEL_DIRECTORY)
    except:
        pass
    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(MODEL_DIRECTORY, 'model.png'))
    print(model.summary())

    # TRAIN model
    history = model.fit(trainSet, epochs=EPOCHS)
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

print('Model evaluation: ', model.evaluate(testSet))
