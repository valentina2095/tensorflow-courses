#########################
#    COMPUTER VISION    #
#########################
import tensorflow as tf
import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    # the last layer must contain as many neuron as categories
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
                   loss='sparse_categorical_crossentropy')


model.fit(x=train_imgs/255, y=train_labels, epochs=15)

model.evaluate(test_imgs/255, test_labels)
