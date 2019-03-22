# full notebook on https://github.com/lmoroney/dlaicourse Part 2 - Lesson 2

import keras
import numpy as np

# the simplest NN with only one layer and one neuron in it
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# the loss function measure how good or bad is the guess, the optimizer figures
# out the next guess

model.compile(optimizer='sgd', loss='mean_squared_error')

X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# model training
model.fit(x=X, y=Y, epochs=500)

# make predictions with the trained model

print(model.predict([10.0]))
