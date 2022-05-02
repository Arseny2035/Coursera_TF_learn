import tensorflow_datasets as tfds
import tensorflow as tf

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cv2
import wget

# Download and Prepare the Dataset
# We will use the Cats vs Dogs dataset and we can load it via Tensorflow Datasets. The images are labeled 0 for cats and 1 for dogs.
train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
validation_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)
test_data = tfds.load('cats_vs_dogs', split='train[-10%:]', as_supervised=True)


# The cell below will preprocess the images and create batches before feeding it to our model.
def augment_images(image, label):
    # cast to float
    image = tf.cast(image, tf.float32)
    # normalize the pixel values
    image = (image / 255)
    # resize to 300 x 300
    image = tf.image.resize(image, (300, 300))

    return image, label


# use the utility function above to preprocess the images
augmented_training_data = train_data.map(augment_images)

# shuffle and create batches before training
train_batches = augmented_training_data.shuffle(1024).batch(32)

# Build the classifier
# This will look familiar to you because it is almost identical to the previous model we built.
# The key difference is the output is just one unit that is sigmoid activated. This is because
# we're only dealing with two classes.
model = Sequential()
model.add(Conv2D(16, input_shape=(300, 300, 3), kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))

model.summary()

# The loss can be adjusted from last time to deal with just two classes.
# For that, we pick binary_crossentropy.
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001))
model.fit(train_batches, epochs=25)

# Building the CAM model
# You will follow the same steps as before in generating the class activation maps.
gap_weights = model.layers[-1].get_weights()[0]
gap_weights.shape

cam_model = Model(inputs=model.input, outputs=(model.layers[-3].output, model.layers[-1].output))
cam_model.summary()


def show_cam(image_value, features, results):
    '''
    Displays the class activation map of an image

    Args:
      image_value (tensor) -- preprocessed input image with size 300 x 300
      features (array) -- features of the image, shape (1, 37, 37, 128)
      results (array) -- output of the sigmoid layer
    '''

    # there is only one image in the batch so we index at `0`
    features_for_img = features[0]
    prediction = results[0]

    # there is only one unit in the output so we get the weights connected to it
    class_activation_weights = gap_weights[:, 0]

    # upsample to the image size
    class_activation_features = sp.ndimage.zoom(features_for_img, (300 / 37, 300 / 37, 1), order=2)

    # compute the intensity of each feature in the CAM
    cam_output = np.dot(class_activation_features, class_activation_weights)

    # visualize the results
    print(f'sigmoid output: {results}')
    print(f"prediction: {'dog' if round(results[0][0]) else 'cat'}")
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
    plt.imshow(tf.squeeze(image_value), alpha=0.5)
    plt.show()


# Testing the Model
# Let's download a few images and see how the class activation maps look like.

wget.download('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/cat1.jpg', 'cat1.jpg')


# !wget -O cat2.jpg https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/cat2.jpg
# !wget -O catanddog.jpg https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/catanddog.jpg
# !wget -O dog1.jpg https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/dog1.jpg
# !wget -O dog2.jpg https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/dog2.jpg


# utility function to preprocess an image and show the CAM
def convert_and_classify(image):
    # load the image
    img = cv2.imread(image)

    # preprocess the image before feeding it to the model
    img = cv2.resize(img, (300, 300)) / 255.0

    # add a batch dimension because the model expects it
    tensor_image = np.expand_dims(img, axis=0)

    # get the features and prediction
    features, results = cam_model.predict(tensor_image)

    # generate the CAM
    show_cam(tensor_image, features, results)


convert_and_classify('cat1.jpg')
# convert_and_classify('cat2.jpg')
# convert_and_classify('catanddog.jpg')
# convert_and_classify('dog1.jpg')
# convert_and_classify('dog2.jpg')

# Let's also try it with some of the test images before we make some observations.
# preprocess the test images
augmented_test_data = test_data.map(augment_images)
test_batches = augmented_test_data.batch(1)

for img, lbl in test_batches.take(5):
    print(f"ground truth: {'dog' if lbl else 'cat'}")
    features, results = cam_model.predict(img)
    show_cam(img, features, results)
