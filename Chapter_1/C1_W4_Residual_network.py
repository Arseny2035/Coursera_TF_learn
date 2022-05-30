import tensorflow as tf
import tensorflow_datasets as tfds
from keras.models import model_from_json
import os

class IdentifyBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(IdentifyBlock, self).__init__(name='')

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

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

class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3))

        self.id1a = IdentifyBlock(64, 3)
        self.id1b = IdentifyBlock(64, 3)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

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
LOAD_MODEL = True

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

MODEL_FILE = 'Data/ResNet/models/model.json'
WEIGHTS_FILE = 'Data/ResNet/models/model_weights.h5'

# Load or create model.
if LOAD_MODEL and os.path.exists(MODEL_FILE) \
        and os.path.exists(WEIGHTS_FILE):

    # Load saved model if it exists.
    json_file = open(MODEL_FILE)
    loaded_model_json = json_file.read()
    json_file.close()
    resnet = model_from_json(loaded_model_json)
    resnet.load_weights(WEIGHTS_FILE)
    resnet.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    print("Model was load from disk!")

else:

    resnet = ResNet(10)
    resnet.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    dataset = tfds.load('mnist', split=tfds.Split.TRAIN)
    dataset = dataset.map(preprocess).batch(32)
    history = resnet.fit(dataset, epochs=1)


    # Save new model if we had to.
    if MUST_SAVE_THE_MODEL:
        # model_json = resnet.to_json()
        # with open(MODEL_FILE, 'w') as json_file:
        #     json_file.write(model_json)
        #     json_file.close()
        resnet.save(MODEL_FILE)

        resnet.save_weights(WEIGHTS_FILE)
        print("Model saved to disk!")











