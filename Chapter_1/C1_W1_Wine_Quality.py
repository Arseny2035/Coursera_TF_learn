import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils

# URL of the white wine dataset
URI = 'data/W1/winequality-white.csv'

# load the dataset from the URL
white_df = pd.read_csv(URI, sep=";")

# fill the `is_red` column with zeros.
white_df["is_red"] = 0

# keep only the first of duplicate items
white_df = white_df.drop_duplicates(keep='first')

utils.test_white_df(white_df)

print(white_df.alcohol[0])
print(white_df.alcohol[100])

# URL of the red wine dataset
URI = 'data/W1/winequality-red.csv'

# load the dataset from the URL
red_df = pd.read_csv(URI, sep=";")

# fill the `is_red` column with ones.
red_df["is_red"] = 1

# keep only the first of duplicate items
red_df = red_df.drop_duplicates(keep='first')

utils.test_red_df(red_df)

print(red_df.alcohol[0])
print(red_df.alcohol[100])


df = pd.concat([red_df, white_df], ignore_index=True)

print(df.alcohol[0])
print(df.alcohol[100])

df['quality'].hist(bins=20)

# get data with wine quality greater than 4 and less than 8
df = df[(df['quality'] > 4) & (df['quality'] < 8)]

# reset index and drop the old one
df = df.reset_index(drop=True)

print(df.alcohol[0])
print(df.alcohol[100])

df['quality'].hist(bins=20)

# split df into 80:20 train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=1)

# split train into 80:20 train and val sets
train, val = train_test_split(train, test_size=0.2, random_state=1)

utils.test_data_sizes(train.size, test.size, val.size)

train_stats = train.describe()
train_stats.pop('is_red')
train_stats.pop('quality')
train_stats = train_stats.transpose()

print(train_stats)

def format_output(data):
    is_red = data.pop('is_red')
    is_red = np.array(is_red)
    quality = data.pop('quality')
    quality = np.array(quality)
    return (quality, is_red)


# format the output of the train set
train_Y = format_output(train)

# format the output of the val set
val_Y = format_output(val)

# format the output of the test set
test_Y = format_output(test)

utils.test_format_output(df, train_Y, val_Y, test_Y)

print(train.head())

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# normalize the train set
norm_train_X = norm(train)

# normalize the val set
norm_val_X = norm(val)

# normalize the test set
norm_test_X = norm(test)

utils.test_norm(norm_train_X, norm_val_X, norm_test_X, train, val, test)


def base_model(inputs):
    # connect a Dense layer with 128 neurons and a relu activation
    x = Dense(128, activation=tf.nn.relu)(inputs)

    # connect another Dense layer with 128 neurons and a relu activation
    x = Dense(128, activation=tf.nn.relu)(x)
    return x

utils.test_base_model(base_model)


def final_model(inputs):
    # get the base model
    x = base_model(inputs)

    # connect the output Dense layer for regression
    wine_quality = Dense(units='1', name='wine_quality')(x)

    # connect the output Dense layer for classification. this will use a sigmoid activation.
    wine_type = Dense(units='1', activation=tf.keras.activations.sigmoid, name='wine_type')(x)

    # define the model using the input and output layers
    model = Model(inputs=inputs, outputs=[wine_quality, wine_type])

    return model

utils.test_final_model(final_model)
##