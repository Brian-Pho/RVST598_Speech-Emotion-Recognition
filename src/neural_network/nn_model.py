"""
This file builds the machine learning model and holds helper functions for the
model.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, backend
from sklearn import metrics

import nn_constants as nnc
from src.database_processor import db_constants as dbc
from src import em_constants as emc


# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def build_model():
    """
    Builds a neural network model to classify emotions in speech.

    :return: keras.model
    """
    model = models.Sequential()

    model.add(layers.Conv2D(96, (3, 3), activation='relu',
                            input_shape=nnc.INPUT_SHAPE,
                            ))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            ))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',))
    model.add(layers.Conv2D(96, (3, 3), activation='relu',))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu',))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu',))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu',))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7, activation='sigmoid'))
    model.compile(optimizer=nnc.OPTIMIZER, loss=nnc.LOSS, metrics=nnc.METRICS)

    # # Print the model to the console
    # model.summary()
    # # Print the model to a png file
    # utils.plot_model(model, show_shapes=True, to_file=nnc.MODEL_PLOT_PATH)
    # # Turn into multi-gpu model
    # model = utils.multi_gpu_model(model, gpus=2)

    return model


def train_model(model, train_gen, valid_gen, class_weights=None):
    """
    Trains a neural network model for speech emotion recognition.

    :param model: The model to train
    :param train_gen: The generator creating training samples
    :param valid_gen: The generator creating validation samples
    :param class_weights: The class weights if the dataset is imbalanced
    :return: keras.History, representing the training history
    """
    start_train_time = time.time()
    history = model.fit_generator(
        generator=train_gen, epochs=nnc.NUM_EPOCHS, verbose=nnc.VERBOSE_LVL,
        validation_data=valid_gen, class_weight=class_weights,
        use_multiprocessing=True, workers=nnc.NUM_WORKERS
    )
    end_train_time = time.time()

    print("Training the model took {} seconds.".format(
        end_train_time - start_train_time))

    return history


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


def visualize_confusion_matrix(model, test_gen):
    """
    Visualizes the test set confusion matrix.

    :param model: The ML model
    :param test_gen: A BatchGenerator object containing the test samples
    """
    true_output = []
    pred_output = []

    for batch_num in range(0, len(test_gen)):
        # Get a batch
        batch_inputs, batch_targets = test_gen[batch_num]
        # Get the neural network's prediction from the batch input
        pred_output.append(model.predict_on_batch(batch_inputs))
        true_output.append(batch_targets)

    # Convert to numpy array
    true_output = np.array(true_output, dtype=int)
    pred_output = np.array(pred_output, dtype=int)

    # Reshape due to batches
    true_output = true_output.reshape(
        true_output.shape[0] * true_output.shape[1], true_output.shape[2])
    pred_output = pred_output.reshape(
        pred_output.shape[0] * pred_output.shape[1], pred_output.shape[2])

    # Get the confusion matrix for each emotion
    confusion_matrix = metrics.multilabel_confusion_matrix(
        true_output, pred_output)

    # Display the confusion matrix for each emotion
    labels = ["0", "1"]

    for emotion in range(0, emc.NUM_EMOTIONS):
        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix[emotion])

        # Create colorbar
        ax.figure.colorbar(im, ax=ax)

        # Set the x and y axis labels
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

        # Set the x and y axis tick values
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))

        # Set the x and y axis tick labels
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Display the value of each in the matrix
        for row in range(confusion_matrix[emotion].shape[0]):
            for col in range(confusion_matrix[emotion].shape[1]):
                ax.text(col, row, confusion_matrix[emotion][row][col],
                        ha="center", va="center", color="w")

        fig.tight_layout()
        plt.show()
