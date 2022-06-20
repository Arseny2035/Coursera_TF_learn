import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
import cv2

data_dir = "../../../tensorflow_datasets/CUB200(2011)/CUB_200_2011/CUB_200_2011"

# 1. Visualization Utilities
#
# 1.1 Bounding Boxes Utilities
# We have provided you with some functions which you will use to draw bounding boxes around the birds in the image.
#
# draw_bounding_box_on_image: Draws a single bounding box on an image.
# draw_bounding_boxes_on_image: Draws multiple bounding boxes on an image.
# draw_bounding_boxes_on_image_array: Draws multiple bounding boxes on an array of images.

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color=(255, 0, 0), thickness=5):
    """
    Adds a bounding box to an image.
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
    """

    image_width = image.shape[1]
    image_height = image.shape[0]
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)


def draw_bounding_boxes_on_image(image, boxes, color=[], thickness=5):
    """
    Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3],
                                   boxes[i, 2], color[i], thickness)


def draw_bounding_boxes_on_image_array(image, boxes, color=[], thickness=5):
    """
    Draws bounding boxes on image (numpy array).

    Args:
      image: a numpy array object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: a list of strings for each bounding box.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """

    draw_bounding_boxes_on_image(image, boxes, color, thickness)

    return image


# 1.2 Data and Predictions Utilities
# We've given you some helper functions and code that are used to visualize the data and the model's predictions.
#
# display_digits_with_boxes: This displays a row of "digit" images along with the model's predictions for each image.
# plot_metrics: This plots a given metric (like loss) as it changes over multiple epochs of training.

# Matplotlib config
plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')  # Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

iou_threshold = 0.5

# utility to display a row of digits with their predictions
def display_digits_with_boxes(images, pred_bboxes, bboxes, iou, title, bboxes_normalized=False):
    n = len(images)

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(n):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        if (len(pred_bboxes) > i):
            bbox = pred_bboxes[i]
            bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1],
                    bbox[3] * images[i].shape[0]]
            bboxes_to_plot.append(bbox)

        if (len(bboxes) > i):
            bbox = bboxes[i]
            if bboxes_normalized == True:
                bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1],
                        bbox[3] * images[i].shape[0]]
            bboxes_to_plot.append(bbox)

        img_to_draw = draw_bounding_boxes_on_image_array(image=images[i], boxes=np.asarray(bboxes_to_plot),
                                                         color=[(255, 0, 0), (0, 255, 0)])
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img_to_draw)

        if len(iou) > i:
            color = "black"
            if (iou[i][0] < iou_threshold):
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" % (iou[i][0]), color=color, transform=ax.transAxes)

    # plt.show()

# utility to display training and validation curves
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)


# 2. Preprocess and Load the Dataset
#
# 2.1 Preprocessing Utilities
# We have given you some helper functions to pre-process the image data.
#
# read_image_tfds
# Resizes image to (224, 224)
# Normalizes image
# Translates and normalizes bounding boxes

def read_image_tfds(image, bbox):
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    image = tf.image.resize(image, (224, 224,))

    image = image / 127.5
    image -= 1

    bbox_list = [bbox[0] / factor_x,
                 bbox[1] / factor_y,
                 bbox[2] / factor_x,
                 bbox[3] / factor_y]

    return image, bbox_list

# read_image_with_shape
# This is very similar to read_image_tfds except it also keeps a copy of the original image (before pre-processing) and returns this as well.
#
# Makes a copy of the original image.
# Resizes image to (224, 224)
# Normalizes image
# Translates and normalizes bounding boxes

def read_image_with_shape(image, bbox):
    original_image = image

    image, bbox_list = read_image_tfds(image, bbox)

    return original_image, image, bbox_list



# dataset_to_numpy_util
# This function converts a dataset into numpy arrays of images and boxes.
#
# This will be used when visualizing the images and their bounding boxes

def dataset_to_numpy_util(dataset, batch_size=0, N=0):
    # eager execution: loop through datasets normally
    take_dataset = dataset.shuffle(1024)

    if batch_size > 0:
        take_dataset = take_dataset.batch(batch_size)

    if N > 0:
        take_dataset = take_dataset.take(N)

    if tf.executing_eagerly():
        ds_images, ds_bboxes = [], []
        for images, bboxes in take_dataset:
            ds_images.append(images.numpy())
            ds_bboxes.append(bboxes.numpy())

    return (np.array(ds_images, dtype="object"), np.array(ds_bboxes, dtype="object"))


# dataset_to_numpy_with_original_bboxes_util
# This function converts a dataset into numpy arrays of
# original images
# resized and normalized images
# bounding boxes
# This will be used for plotting the original images with true and predicted bounding boxes.

def dataset_to_numpy_with_original_bboxes_util(dataset, batch_size=0, N=0):
    normalized_dataset = dataset.map(read_image_with_shape)
    if batch_size > 0:
        normalized_dataset = normalized_dataset.batch(batch_size)

    if N > 0:
        normalized_dataset = normalized_dataset.take(N)

    if tf.executing_eagerly():
        ds_original_images, ds_images, ds_bboxes = [], [], []

    for original_images, images, bboxes in normalized_dataset:
        ds_images.append(images.numpy())
        ds_bboxes.append(bboxes.numpy())
        ds_original_images.append(original_images.numpy())

    return np.array(ds_original_images), np.array(ds_images), np.array(ds_bboxes)

# read_image_tfds_with_original_bbox
# This function reads image from data
# It also denormalizes the bounding boxes (it undoes the bounding box normalization that is performed by the previous two helper functions.)

def read_image_tfds_with_original_bbox(data):
    image = data["image"]
    bbox = data["bbox"]

    shape = tf.shape(image)
    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    bbox_list = [bbox[1] * factor_x ,
                 bbox[0] * factor_y,
                 bbox[3] * factor_x,
                 bbox[2] * factor_y]
    return image, bbox_list

# 2.2 Visualize the images and their bounding box labels
# Now you'll take a random sample of images from the training and validation sets and visualize
# them by plotting the corresponding bounding boxes.
#
# Visualize the training images and their bounding box labels

##############################################
# Errors in downloaded dataset
# def get_visualization_training_dataset():
#     try:
#         os.makedirs(data_dir)
#     except:
#         pass
#     print(data_dir)
#     dataset, info = tfds.load("caltech_birds2010", split="train", data_dir=data_dir, with_info=True, download=True)
#     print(info)
#     visualization_training_dataset = dataset.map(read_image_tfds_with_original_bbox,
#                                                  num_parallel_calls=16)
#     return visualization_training_dataset
###############################################



def get_visualization_training_dataset():

    print(data_dir)
    use flow_from_directory of somathing else...
    Image... download..\
    dataset = tfds.load("caltech_birds2010", split="train", data_dir=data_dir, with_info=True, download=True)
    bboxes = None


    visualization_training_dataset = dataset.map(read_image_tfds_with_original_bbox,
                                                 num_parallel_calls=16)
    return visualization_training_dataset

visualization_training_dataset = get_visualization_training_dataset()
print("Dataset is loaded")

(visualization_training_images, visualization_training_bboxes) = \
    dataset_to_numpy_util(visualization_training_dataset, N=10)
display_digits_with_boxes(np.array(visualization_training_images), np.array([]),
                          np.array(visualization_training_bboxes), np.array([]),
                          "training images and their bboxes")

# Visualize the validation images and their bounding boxes
def get_visualization_validation_dataset():
    dataset = tfds.load("caltech_birds2010", split="test", data_dir=data_dir, download=True)
    visualization_validation_dataset = dataset.map(read_image_tfds_with_original_bbox,
                                                   num_parallel_calls=16)
    return visualization_validation_dataset


visualization_validation_dataset = get_visualization_validation_dataset()

(visualization_validation_images, visualization_validation_bboxes) = \
    dataset_to_numpy_util(visualization_validation_dataset, N=10)
display_digits_with_boxes(np.array(visualization_validation_images), np.array([]),
                          np.array(visualization_validation_bboxes), np.array([]),
                          "validation images and their bboxes")


# 2.3 Load and prepare the datasets for the model
# These next two functions read and prepare the datasets that you'll feed to the model.
#
# They use read_image_tfds to resize, and normalize each image and its bounding box label.
# They performs shuffling and batching.
# You'll use these functions to create training_dataset and validation_dataset,
# which you will give to the model that you're about to build.

BATCH_SIZE = 64

def get_training_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(-1)
    return dataset

def get_validation_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset

training_dataset = get_training_dataset(visualization_training_dataset)
validation_dataset = get_validation_dataset(visualization_validation_dataset)


# 3. Define the Network
# Bounding box prediction is treated as a "regression" task, in that you want the model
# to output numerical values.
#
# You will be performing transfer learning with MobileNet V2. The model architecture
# is available in TensorFlow Keras.
# You'll also use pretrained 'imagenet' weights as a starting point for further training.
# These weights are also readily available
# You will choose to retrain all layers of MobileNet V2 along with the final classification layers.
# Note: For the following exercises, please use the TensorFlow Keras Functional API
# (as opposed to the Sequential API).

# Exercise 1
# Please build a feature extractor using MobileNetV2.
#
# First, create an instance of the mobilenet version 2 model
#
# Please check out the documentation for MobileNetV2
# Set the following parameters:
# input_shape: (height, width, channel): input images have height and width of 224 by 224,
# and have red, green and blue channels.
# include_top: you do not want to keep the "top" fully connected layer, since you will customize
# your model for the current task.
# weights: Use the pre-trained 'imagenet' weights.
# Next, make the feature extractor for your specific inputs by passing the inputs into your
# mobilenet model.
#
# For example, if you created a model object called some_model and have inputs stored in x,
# you'd invoke the model and pass in your inputs like this: some_model(x) to get the feature
# extractor for your given inputs x.
# Note: please use mobilenet_v2 and not mobile_net or mobile_net_v3

def feature_extractor(inputs):
    ### YOUR CODE HERE ###

    # Create a mobilenet version 2 model object
    mobilenet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), include_top=False,
                                                                     weights='imagenet')

    # pass the inputs into this modle object to get a feature extractor for these inputs
    feature_extractor = mobilenet_model(inputs)

    ### END CODE HERE ###

    # return the feature_extractor
    return feature_extractor


# Exercise 2
# Next, you'll define the dense layers to be used by your model.
#
# You'll be using the following layers
#
# GlobalAveragePooling2D: pools the features.
# Flatten: flattens the pooled layer.
# Dense: Add two dense layers:
# A dense layer with 1024 neurons and a relu activation.
# A dense layer following that with 512 neurons and a relu activation.
# Note: Remember, please build the model using the Functional API syntax (as opposed
# to the Sequential API).

def dense_layers(features):
    ### YOUR CODE HERE ###

    # global average pooling 2d layer
    x = tf.keras.layers.GlobalAveragePooling2D()(features)

    # flatten layer
    x = tf.keras.layers.Flatten()(x)

    # 1024 Dense layer, with relu
    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)

    # 512 Dense layer, with relu
    x = tf.keras.layers.Dense(units=512, activation='relu')(x)

    ### END CODE HERE ###

    return x

# Exercise 3
# Now you'll define a layer that outputs the bounding box predictions.
#
# You'll use a Dense layer.
# Remember that you have 4 units in the output layer, corresponding to (xmin, ymin, xmax, ymax).
# The prediction layer follows the previous dense layer, which is passed into this function
# as the variable x.
# For grading purposes, please set the name parameter of this Dense layer to be `bounding_box'

def bounding_box_regression(x):
    ### YOUR CODE HERE ###

    # Dense layer named `bounding_box`
    bounding_box_regression_output = tf.keras.layers.Dense(units=4, name='bounding_box')(x)

    ### END CODE HERE ###

    return bounding_box_regression_output

# Exercise 4
# Now, you'll use those functions that you have just defined above to construct the model.
#
# feature_extractor(inputs)
# dense_layers(features)
# bounding_box_regression(x)
# Then you'll define the model object using Model. Set the two parameters:
# inputs
# outputs

def final_model(inputs):
    ### YOUR CODE HERE ###

    # features
    feature_cnn = feature_extractor(inputs)

    # dense layers
    last_dense_layer = dense_layers(feature_cnn)

    # bounding box
    bounding_box_output = bounding_box_regression(last_dense_layer)

    # define the TensorFlow Keras model using the inputs and outputs to your model
    model = tf.keras.Model(inputs=inputs, outputs=bounding_box_output)

    ### END CODE HERE ###

    return model

# Exercise 5
# Define the input layer, define the model, and then compile the model.
#
# inputs: define an Input layer
# Set the shape parameter. Check your definition of feature_extractor to see the expected
# dimensions of the input image.
# model: use the final_model function that you just defined to create the model.
# compile the model: Check the Model documentation for how to compile the model.
# Set the optimizer parameter to Stochastic Gradient Descent using SGD
# When using SGD, set the momentum to 0.9 and keep the default learning rate.
# Set the loss function of SGD to mean squared error (see the SGD documentation for an example
# of how to choose mean squared error loss).

def define_and_compile_model():
    ### YOUR CODE HERE ###

    # define the input layer
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    # create the model
    model = final_model(inputs)

    # compile your model
    model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    ### END CODE HERE ###

    return model


# define your model
model = define_and_compile_model()
# print model layers
model.summary()


# Train the Model
#
# 4.1 Prepare to Train the Model
# You'll fit the model here, but first you'll set some of the parameters that go into fitting the model.
#
# EPOCHS: You'll train the model for 50 epochs
#
# BATCH_SIZE: Set the BATCH_SIZE to an appropriate value. You can look at the ungraded labs from
# this week for some examples.
#
# length_of_training_dataset: this is the number of training examples. You can find this value by
# getting the length of visualization_training_dataset.
#
# Note: You won't be able to get the length of the object training_dataset. (You'll get an error message).
# length_of_validation_dataset: this is the number of validation examples. You can find this value by
# getting the length of visualization_validation_dataset.
#
# Note: You won't be able to get the length of the object validation_dataset.
# steps_per_epoch: This is the number of steps it will take to process all of the training data.
#
# If the number of training examples is not evenly divisible by the batch size, there will be
# one last batch that is not the full batch size.
# Try to calculate the number steps it would take to train all the full batches plus one more batch
# containing the remaining training examples. There are a couples ways you can calculate this.
# You can use regular division / and import math to use math.ceil() Python math module docs
# Alternatively, you can use // for integer division, % to check for a remainder after integer division,
# and an if statement.
# validation_steps: This is the number of steps it will take to process all of the validation data.
# You can use similar calculations that you did for the step_per_epoch, but for the validation dataset.

# You'll train 50 epochs
EPOCHS = 50

### START CODE HERE ###

# Choose a batch size
BATCH_SIZE = 32

# Get the length of the training set
length_of_training_dataset = len(visualization_training_dataset)

# Get the length of the validation set
length_of_validation_dataset = len(visualization_validation_dataset)

# Get the steps per epoch (may be a few lines of code)
import math
steps_per_epoch = math.ceil(length_of_training_dataset / BATCH_SIZE)

# get the validation steps (per epoch) (may be a few lines of code)
# validation_steps = length_of_validation_dataset//BATCH_SIZE
# if length_of_validation_dataset % BATCH_SIZE > 0:
#     validation_steps += 1
validation_steps = math.ceil(length_of_validation_dataset / BATCH_SIZE)


### END CODE HERE

# 4.2 Fit the model to the data
# Check out the parameters that you can set to fit the Model. Please set the following parameters.
#
# x: this can be a tuple of both the features and labels, as is the case here when using a tf.Data dataset.
# Please use the variable returned from get_training_dataset().
# Note, don't set the y parameter when the x is already set to both the features and labels.
# steps_per_epoch: the number of steps to train in order to train on all examples in the training dataset.
# validation_data: this is a tuple of both the features and labels of the validation set.
# Please use the variable returned from get_validation_dataset()
# validation_steps: teh number of steps to go through the validation set, batch by batch.
# epochs: the number of epochs.
# If all goes well your model's training will start.


### YOUR CODE HERE ####

# Fit the model, setting the parameters noted in the instructions above.
history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, validation_data=validation_dataset,
                    validation_steps=validation_steps, epochs=EPOCHS)

### END CODE HERE ###

# 5. Validate the Model
#
# 5.1 Loss
# You can now evaluate your trained model's performance by checking its loss value on the validation set.

loss = model.evaluate(validation_dataset, steps=validation_steps)
print("Loss: ", loss)

# 5.2 Save your Model for Grading
# When you have trained your model and are satisfied with your validation loss, please you save
# your model so that you can upload it to the Coursera classroom for grading.

# Please save your model
model.save("birds.h5")

# And download it using this shortcut or from the "Files" panel to the left
from google.colab import files

files.download("birds.h5")