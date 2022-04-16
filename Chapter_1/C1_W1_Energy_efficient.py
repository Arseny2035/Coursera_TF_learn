import tensorflow as tf
from tensorflow import keras
from keras import Model, Input
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd



def format_output(data):
    y1 = data.pop('Y1')
    y1 = np.array(y1)
    y2 = data.pop('Y2')
    y2 = np.array(y2)
    return y1, y2

def norm(x):
    print("train_stats['mean'] = ", train_stats['mean'])
    print("train_stats['std'] = ", train_stats['std'])
    print("(x - train_stats['mean']) / train_stats['std'] = ", (x - train_stats['mean']) / train_stats['std'])
    return (x - train_stats['mean']) / train_stats['std']

def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()

def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', labtl='val_' + metric_name)
    plt.show()

# Specify data
URI = 'data/W1/ENB2012_data.xlsx'

# Use pandas excel reader
df = pd.read_excel(URI)
print("df :", df)
df = df.sample(frac=1).reset_index(drop=True)
print("df.sample(frac=1).reset_index(drop=True): ", df)

# Split the data into train and test 80/20
train, test = train_test_split(df, test_size=0.2)
print("train: ", train)
train_stats = train.describe()

# GEt Y1 and Y2 as the 2 outputs and format them as np arrays
train_stats.pop('Y1')
train_stats.pop('Y2')
print("train_stats: ", train_stats)
train_stats = train_stats.transpose()
print("train_stats transpose: ", train_stats)

train_Y = format_output(train)
# print("train_Y: ", train_Y)
test_Y = format_output(test)

# Normalize the training and test data
norm_train_X = norm(train)
norm_test_X = norm(test)



# Define model
input_layer = Input(shape=(len(train.columns),))
first_dense = layers.Dense(units='128', activation='relu')(input_layer)
second_dense = layers.Dense(units='128', activation='relu')(first_dense)

y1_output = layers.Dense(units='1', name='y1_output')(second_dense)
third_dense = layers.Dense(units='64', activation='relu')(second_dense)

y2_output = layers.Dense(units='1', name='y2_output')(third_dense)

model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

print(model.summary())

# Specify the optimizer, compile model with loss function for both outputs
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'y1_output': 'mse', 'y2_output': 'mse'},
              metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                       'y2_output': tf.keras.metrics.RootMeanSquaredError()})

# Train the model
history = model.fit(norm_train_X, train_Y,
                    epochs=500, batch_size=10, validation_data=(norm_test_X, test_Y))

# Test the model and print loss and mse for both outputs
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_Y)
print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".
      format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))

# Plot the loss and mse
Y_pred = model.predict(norm_test_X)
plot_diff(test_Y[0], Y_pred[0], title='Y1')
plot_diff(test_Y[1], Y_pred[1], title='Y2')
plot_metrics(metric_name='y1_output_root_MSE', title='Y1 RMSE', ylim=6)
plot_metrics(metric_name='y2_output_root_MSE', title='Y2 RMSE', ylim=7)
