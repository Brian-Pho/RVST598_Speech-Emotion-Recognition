"""
This file builds the machine learning model.
"""

import matplotlib.pyplot as plt
from keras import layers, models
from keras.utils import plot_model


def build_model():
    """
    Build a CNN to classify emotions in speech using Keras.

    :return:
    """
    model = models.Sequential()
    # kernel_regularizer=regularizers.l2(0.001),
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 163, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    # model.add(layers.Dropout(0.5))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    # plot_model(model, to_file="model.png")

    return model


def display_history(history):
    """
    Displays a neural network's training history.

    :param history: Dictionary from training a neural network
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
