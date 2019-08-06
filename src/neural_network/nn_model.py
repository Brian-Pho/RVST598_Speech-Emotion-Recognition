"""
This file builds the machine learning model and holds helper functions for the
model.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from keras import layers, models, backend, utils, regularizers
from sklearn import metrics

import nn_constants as nnc
from src.database_processor import db_constants as dbc

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def build_model():
    """
    Builds a CNN model to classify emotions in speech using Keras.

    :return: keras.model
    """
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=nnc.INPUT_SHAPE,
                            ))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            ))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu',))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu',))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu',))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7, activation='softmax'))
    model.compile(optimizer=nnc.OPTIMIZER, loss=nnc.LOSS, metrics=nnc.METRICS)

    # # Print the model to the console
    # model.summary()
    # # Print the model to a png file
    # utils.plot_model(model, show_shapes=True, to_file=nnc.MODEL_PLOT_PATH)
    # # Turn into multi-gpu model
    # model = utils.multi_gpu_model(model, gpus=2)

    return model


def save_history(history):
    """
    Saves the training history of a model to text files.

    :param history: Keras.History object
    """
    cat_acc = history.history['categorical_accuracy']
    val_cat_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    np.savetxt(
        os.path.join(dbc.MODEL_PATH, "cat_acc.txt"), cat_acc, delimiter=",")
    np.savetxt(
        os.path.join(dbc.MODEL_PATH, "val_cat_acc.txt"), val_cat_acc,
        delimiter=",")
    np.savetxt(os.path.join(dbc.MODEL_PATH, "loss.txt"), loss, delimiter=",")
    np.savetxt(
        os.path.join(dbc.MODEL_PATH, "val_loss.txt"), val_loss, delimiter=",")


def visualize_train_history(history):
    """
    Visualizes a neural network's training history.

    :param history: Dictionary from training a neural network
    """
    cat_acc = history.history['categorical_accuracy']
    val_cat_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(cat_acc) + 1)

    plt.plot(epochs, cat_acc, 'bo', label='Training cat_acc')
    plt.plot(epochs, val_cat_acc, 'b', label='Validation cat_acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def visualize_interm_activations(model, test_img):
    """
    Visualizes the intermediate activations of each convolution layer.

    :param model: The ML model
    :param test_img: The image to get activations on
    """
    # Extract the outputs of the top three conv layers
    layer_names = [layer.name for layer in model.layers[0:5:2]]
    layer_outputs = [layer.output for layer in model.layers[0:5:2]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(test_img)

    # # To visualize one channel for one activation layer
    # first_layer_activation = activations[0]
    # print(first_layer_activation.shape)
    # plt.pcolormesh(first_layer_activation[0, :, :, 4])
    # plt.show()

    # To visualize all channels for one activation layer
    for name, activation in zip(layer_names, activations):
        for i, j in zip(range(1, 33), range(0, 32)):
            plt.subplot(4, 8, i)
            plt.pcolormesh(activation[0, :, :, j], cmap="magma")

        # plt.title(name)
        plt.show()


def visualize_heatmap_activation(model, test_img):
    """
    Visualizes the heatmap activations of each convolution layer.

    :param model: The ML model
    :param test_img: The image to get activations on
    """
    preds = model.predict(test_img)

    # Setting up the Grad-CAM algorithm
    img_output = model.output[:, np.argmax(preds[0])]
    last_conv_layer = model.get_layer('conv2d_3')
    grads = backend.gradients(img_output, last_conv_layer.output)[0]
    pooled_grads = backend.mean(grads, axis=(0, 1, 2))

    iterate = backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([test_img])

    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # Heatmap post-processing
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Display the test image alongside the heatmap
    plt.pcolormesh(test_img[0, :, :, 0])
    plt.figure()
    plt.pcolormesh(heatmap, cmap="magma")
    plt.show()


def visualize_confusion_matrix(out_true, out_pred):
    """
    Visualizes the test set confusion matrix.

    :param out_true: The ground-truth labels
    :param out_pred: The labels predicted by the neural network
    """
    confusion_matrix = metrics.multilabel_confusion_matrix(out_true, out_pred)
    sn.heatmap(confusion_matrix[0])
    plt.show()
