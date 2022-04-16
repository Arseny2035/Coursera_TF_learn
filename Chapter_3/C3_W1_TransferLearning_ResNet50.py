import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import keras.applications.resnet
import numpy as np
from keras import layers

import tensorflow as tf
from keras.applications.resnet import ResNet50
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

print("tensorflow version: ", tf.__version__)

BATCH_SIZE = 32
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# @title Visualization Utilities[RUN ME]
# Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')  # Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")


# utility to display a row of digits with their predictions
def display_images(digits, predictions, labels, title):
    n = 10

    indexes = np.random.choice(len(predictions), size=n)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_predictions = n_predictions.reshape((n,))
    n_labels = labels[indexes]

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1)
        class_index = n_predictions[i]

        plt.xlabel(classes[class_index])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(n_digits[i])


# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)


# Loading and Preprocessing Data
# CIFAR-10 dataset has 32 x 32 RGB images belonging to 10 classes. You will load the dataset from Keras.
(training_images, training_labels), (validation_images, validation_labels) = \
    tf.keras.datasets.cifar10.load_data()

# Visualize Dataset
# Use the display_image to view some of the images and their class labels.
display_images(training_images, training_labels, training_labels, "Training Data")
display_images(validation_images, validation_labels, validation_labels, "Training Data")


# Preprocess Dataset
# Here, you'll perform normalization on images in training and validation set.
# You'll use the function preprocess_input from the ResNet50 model in Keras.
def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    output_images = keras.applications.resnet.preprocess_input(input_images)
    return output_images


train_X = preprocess_image_input(training_images)
valid_X = preprocess_image_input(validation_images)

# Define the Network
# You will be performing transfer learning on ResNet50 available in Keras.
#
# You'll load pre-trained imagenet weights to the model.
# You'll choose to retain all layers of ResNet50 along with the final classification layers.
'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
Input size is 224 x 224.
'''


def feature_extractor(inputs):
    feature_extractor = keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                           include_top=False,
                                                           weights='imagenet')(inputs)
    return feature_extractor


'''
Defines final dense layers and subsequent softmax layer for classification.
'''


def classifier(inputs):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(10, activation='softmax', name='classifier')(x)
    return x


'''
Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
Connect the feature extraction and "classifier" layers to build the model.
'''


def final_model(inputs):
    resize = layers.UpSampling2D(size=(7, 7))(inputs)

    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output


'''
Define the model and compile it. 
Use Stochastic Gradient Descent as the optimizer.
Use Sparse Categorical CrossEntropy as the loss function.
'''


def define_compile_model():
    inputs = layers.Input(shape=(32, 32, 3))

    classification_output = final_model(inputs)
    model = keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD',
                  loss='sparce_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = define_compile_model()

model.summary()

EPOCHS = 4
history = model.fit(train_X, training_labels, epochs=EPOCHS,
                    validation_data=(valid_X, validation_labels), batch_size=64)



