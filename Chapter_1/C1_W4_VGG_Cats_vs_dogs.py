import tensorflow as tf
import tensorflow_datasets as tfds
import utils
import os
from keras.models import Model, model_from_json, load_model
from keras.utils.vis_utils import plot_model


# Examples:
##################################################
# class MyClass:
#     def __init__(self):
#         self.var1 = 1
#
# my_obj = MyClass()

# print('my_obj.__dict__:', my_obj.__dict__)
# print('vars(my_obj):', vars(my_obj))

# my_obj.var2 = 2
# print('vars(my_obj):', vars(my_obj))

# vars(my_obj)['var3'] = 3
# print('vars(my_obj):', vars(my_obj))

# for i in range(4, 10):
#     vars(my_obj)[f'var{i}'] = 0
# print('vars(my_obj):', vars(my_obj))
##################################################

# Define preprocessing function
def preprocess(features):
    # Resize and normalize
    image = tf.image.resize(features['image'], (224, 224))
    return tf.cast(image, tf.float32) / 255., features['label']


class Block(Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions

        # Define a conv2D_0, conv2D_1, etc based on the number of repetitions
        for i in range(0, repetitions):
            # Define a Conv2D layer, specifying filters, kernel_size, activation and padding.
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')

        # Define the max pool layer that will be added after the Conv2D blocks
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides)

    def call(self, inputs):
        # access the class's conv2D_0 layer
        conv2D_0 = vars(self)['conv2D_0']

        # Connect the conv2D_0 layer to inputs
        x = conv2D_0(inputs)

        # for the remaining conv2D_i layers from 1 to `repetitions` they will be connected to the previous layer
        for i in range(1, self.repetitions):
            # access conv2D_i by formatting the integer `i`. (hint: check how these were saved using `vars()` earlier)
            conv2D_i = vars(self)[f'conv2D_{i}']

            # Use the conv2D_i and connect it to the previous layer
            x = conv2D_i(x)

        # Finally, add the max_pool layer
        max_pool = self.max_pool(x)

        return max_pool


class MyVGG(Model):

    def __init__(self, num_classes):
        super(MyVGG, self).__init__()

        # Creating blocks of VGG with the following
        # (filters, kernel_size, repetitions) configurations
        self.block_a = Block(filters=64, kernel_size=3, repetitions=2)
        self.block_b = Block(filters=128, kernel_size=3, repetitions=2)
        self.block_c = Block(filters=256, kernel_size=3, repetitions=3)
        self.block_d = Block(filters=512, kernel_size=3, repetitions=3)
        self.block_e = Block(filters=512, kernel_size=3, repetitions=3)

        # Classification head
        # Define a Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # Create a Dense layer with 256 units and ReLU as the activation function
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        # Finally add the softmax classifier using a Dense layer
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Chain all the layers one after the other
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


# Set options for model save/load.
MUST_SAVE_THE_MODEL = True
MUST_LOAD_MODEL = True

# SAVE_LOAD_MODEL_METHOD = 'model.to_json'
SAVE_LOAD_MODEL_METHOD = 'model.save'

MODEL_DIRECTORY = 'Data/VGG_Cats_vs_dogs/Models'
MODEL_FILE = os.path.join(MODEL_DIRECTORY, 'saved_model.pb')
WEIGHTS_FILE = os.path.join(MODEL_DIRECTORY, 'model_weights.h5')

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

EPOCHS = 1
BATCH_SIZE = 32

# Create DATASET
setattr(tfds.image_classification.cats_vs_dogs, '_URL',
        'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip')
splits = ['train[:20%]', 'train[20%:30%]']
train_dataset, test_dataset = tfds.load('cats_vs_dogs',
                                        split=splits, data_dir='../data/cats_vs_dogs')


# Apply transformations to dataset
train_dataset = train_dataset.map(preprocess).batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess).batch(BATCH_SIZE)

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

else:
    # Creating model

    # Initialize VGG with the number of classes
    model = MyVGG(num_classes=2)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    print("Model was created!")

    try:
        os.makedirs(MODEL_DIRECTORY)
    except:
        pass
    # plot_model(model, show_shapes=True, show_layer_names=True,
    #            to_file=os.path.join(MODEL_DIRECTORY, 'model.png'))
    # print(model.summary())

    # TRAIN model
    history = model.fit(train_dataset, epochs=EPOCHS)
    print('history: ', history)

    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(MODEL_DIRECTORY, 'model.png'))
    print(model.summary())

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
