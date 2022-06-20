import tensorflow as tf
from tensorflow import keras
from keras import Model, Input
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from keras.models import model_from_json

# Specify data
URI = 'Data/Energy_efficient_dense/ENB2012_data.xlsx'

# Use pandas excel reader
df = pd.read_excel(URI)
print("df :", df)
df = df.sample(frac=1).reset_index(drop=True)
print("df.sample(frac=1).reset_index(drop=True): ", df)

# Split the data into train and test 80/20
train, test = train_test_split(df, test_size=0.2)
print("train: ", train)
train_stats = train.describe()

print(train.columns)
