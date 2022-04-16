from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt

splits, info = tfds.load('horses_or_humans', as_supervised=True, with_info=True,
                         split=['train[:80%]', 'train[80%:]', 'test'], data_dir='../data')

(train_examples, validation_examples, test_examples) = splits

print("info: ", info)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
print("num_examples: ", num_examples)
print("num_classes: ", num_classes)
print("info.features['label']: ", info.features['label'])

BATCH_SIZE = 32
IMAGE_SIZE = 224

# Pre-process an image (please complete this section)
# You'll define a mapping function that resizes the image to a height of 224 by 224, and normalizes the pixels to the range of 0 to 1. Note that pixels range from 0 to 255.
#
# You'll use the following function: tf.image.resize and pass in the (height,width) as a tuple (or list).
# To normalize, divide by a floating value so that the pixel range changes from [0,255] to [0,1].

# Create a autograph pre-processing function to resize and normalize an image
### START CODE HERE ###
@tf.function
def map_fn(img, label):
    image_height = 224
    image_width = 224
### START CODE HERE ###
    # resize the image
    img = tf.image.resize(img, (image_height, image_width))
    # normalize the image
    img /= 255.0
### END CODE HERE
    return img, label

# Apply pre-processing to the datasets (please complete this section)
# Apply the following steps to the training_examples:
#
# Apply the map_fn to the training_examples
# Shuffle the training data using .shuffle(buffer_size=) and set the buffer size to the number of examples.
# Group these into batches using .batch() and set the batch size given by the parameter.
# Hint: You can look at how validation_examples and test_examples are pre-processed to get a sense of how to chain together multiple function calls.

# Prepare train dataset by using preprocessing with map_fn, shuffling and batching
def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    ### START CODE HERE ###
    train_ds = train_examples.map(map_fn)
    train_ds = train_ds.shuffle(buffer_size=num_examples).batch(batch_size)
    ### END CODE HERE ###
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)

    return train_ds, valid_ds, test_ds

train_ds, valid_ds, test_ds = prepare_dataset(train_examples, validation_examples,
                                              test_examples, num_examples, map_fn, BATCH_SIZE)

MODULE_HANDLE = 'data/resnet_50_feature_vector'
model = tf.keras.Sequential([
    hub.KerasLayer(MODULE_HANDLE, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.summary()


def set_adam_optimizer():
    ### START CODE HERE ###
    # Define the adam optimizer
    optimizer = tf.keras.optimizers.Adam()
    ### END CODE HERE ###
    return optimizer


def set_sparse_cat_crossentropy_loss():
    ### START CODE HERE ###
    # Define object oriented metric of Sparse categorical crossentropy for train and val loss
    train_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    val_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    ### END CODE HERE ###
    return train_loss, val_loss

def set_sparse_cat_crossentropy_accuracy():
    ### START CODE HERE ###
    # Define object oriented metric of Sparse categorical accuracy for train and val accuracy
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    ### END CODE HERE ###
    return train_accuracy, val_accuracy

optimizer = set_adam_optimizer()
train_loss, val_loss = set_sparse_cat_crossentropy_loss()
train_accuracy, val_accuracy = set_sparse_cat_crossentropy_accuracy()

# Define the training loop (please complete this section)
# In the training loop:
#
# Get the model predictions: use the model, passing in the input x
# Get the training loss: Call train_loss, passing in the true y and the predicted y.
# Calculate the gradient of the loss with respect to the model's variables: use tape.gradient and pass in the loss and the model's trainable_variables.
# Optimize the model variables using the gradients: call optimizer.apply_gradients and pass in a zip() of the two lists: the gradients and the model's trainable_variables.
# Calculate accuracy: Call train_accuracy, passing in the true y and the predicted y.


# this code uses the GPU if available, otherwise uses a CPU
device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
EPOCHS = 2


# Custom training step
def train_one_step(model, optimizer, x, y, train_loss, train_accuracy):
    '''
    Trains on a batch of images for one step.

    Args:
        model (keras Model) -- image classifier
        optimizer (keras Optimizer) -- optimizer to use during training
        x (Tensor) -- training images
        y (Tensor) -- training labels
        train_loss (keras Loss) -- loss object for training
        train_accuracy (keras Metric) -- accuracy metric for training
    '''
    with tf.GradientTape() as tape:
        ### START CODE HERE ###
        # Run the model on input x to get predictions
        predictions = model(x)
        # Compute the training loss using `train_loss`, passing in the true y and the predicted y
        loss = train_loss(y_true=y, y_pred=predictions)

    # Using the tape and loss, compute the gradients on model variables using tape.gradient
    grads = tape.gradient(loss, model.trainable_variables)

    # Zip the gradients and model variables, and then apply the result on the optimizer
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Call the train accuracy object on ground truth and predictions
    train_accuracy(y, predictions)
    ### END CODE HERE
    return loss

# Define the 'train' function (please complete this section)
# You'll first loop through the training batches to train the model. (Please complete these sections)
#
# The train function will use a for loop to iteratively call the train_one_step function that you just defined.
# You'll use tf.print to print the step number, loss, and train_accuracy.result() at each step. Remember to use tf.print when you plan to generate autograph code.
# Next, you'll loop through the batches of the validation set to calculation the validation loss and validation accuracy. (This code is provided for you). At each iteration of the loop:
#
# Use the model to predict on x, where x is the input from the validation set.
# Use val_loss to calculate the validation loss between the true validation 'y' and predicted y.
# Use val_accuracy to calculate the accuracy of the predicted y compared to the true y.
# Finally, you'll print the validation loss and accuracy using tf.print. (Please complete this section)
#
# print the final loss, which is the validation loss calculated by the last loop through the validation dataset.
# Also print the val_accuracy.result().
# HINT If you submit your assignment and see this error for your stderr output:
#
# Cannot convert 1e-07 to EagerTensor of dtype int64
# Please check your calls to train_accuracy and val_accuracy to make sure that you pass in the true and predicted values in the correct order (check the documentation to verify the order of parameters).


