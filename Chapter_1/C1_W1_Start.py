from keras.layers import Dense, Flatten, Input
from keras import Model

input = Input(shape=(28, 28))

x = Flatten()(input)
x = Dense(128, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

funk_model = Model(inputs=input, outputs=predictions)