"""
This file holds functions to generate data for a machine learning model.
"""

import os

import numpy as np
from keras.utils import Sequence

from src import em_constants as emc
from src.audio_processor import au_constants as auc
from src.database_processor import db_constants as dbc
from src.neural_network import nn_constants as nnc

EMO_INDEX = 2


class BatchGenerator(Sequence):
    """
    Generates a batch used for neural network training.
    """

    def __init__(self, sample_fns, batch_size=nnc.BATCH_SIZE,
                 dim=auc.MEL_SPECGRAM_SHAPE, n_channels=nnc.NUM_CHANNELS,
                 n_classes=emc.NUM_EMOTIONS, shuffle=True):
        """
        Class initializer.

        :param sample_fns: The list of sample filenames to train on
        :param batch_size: The size of a batch
        :param dim: The dimensions of the input data
        :param n_channels: The number of channels in the input data
        :param n_classes: The number of classes in the target data
        :param shuffle: Whether to shuffle the list of sample filenames after
                        each epoch
        """
        self.sample_fns = sample_fns
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, index):
        """
        Gets a batch for neural network training.

        :param index: The index of a batch
        :return: Tuple, consisting of inputs and targets
        """
        batch_start = index * self.batch_size
        batch_end = (index + 1) * self.batch_size

        batch_sample_fns = self.sample_fns[batch_start:batch_end]
        batch_inputs, batch_targets = self._create_batch(batch_sample_fns)

        return batch_inputs, batch_targets

    def __len__(self):
        """
        Gets the numbers of batches per epoch.

        :return: Int, representing how many batches fit into one epoch
        """
        return int(np.ceil(len(self.sample_fns) / self.batch_size))

    def on_epoch_end(self):
        """
        At the end of an epoch, shuffle the data to prevent batches from being
        the same across epochs.
        """
        if self.shuffle:
            np.random.shuffle(self.sample_fns)

    def _create_batch(self, batch_fns):
        """
        Creates a batch given a list of sample filenames.

        :param batch_fns: A list of sample filenames
        :return: Tuple consisting of inputs and targets
        """
        batch_inputs = np.empty((self.batch_size, *self.dim, self.n_channels))
        batch_targets = np.empty((self.batch_size, emc.NUM_EMOTIONS), dtype=int)

        for index, sample_fn in enumerate(batch_fns):
            # Load a sample
            sample_path = os.path.join(dbc.PROCESS_DB_PATH, sample_fn)
            sample = np.load(sample_path, allow_pickle=False)
            # Add a dimension because images must be 3D where the last dimension
            # is the number of channels. In this case there's only one channel.
            sample = np.expand_dims(sample, axis=3)

            # Load a label
            label = read_label(sample_fn)

            batch_inputs[index] = sample
            batch_targets[index] = label

        return batch_inputs, batch_targets


def get_sample_filenames():
    """
    Gets the sample filenames in the processed data folder. Doesn't
    append the path to save memory.

    :return: List
    """
    return os.listdir(dbc.PROCESS_DB_PATH)


def read_label(filename):
    """
    Given a filename, it returns an integer representing the emotion label of
    the file/sample.

    Sample input:
        "CRE_0-0_1_0_0_0_0_0.npy"

    Sample output:
        [0., 1., 0., 0., 0., 0., 0.]

    :return: np.array
    """
    # Remove the file extension
    filename = os.path.splitext(filename)[0]
    # Parse the label
    k_hot_encoded_label = filename.split("-")[1].split(b"_")
    # Convert into a numpy array
    k_hot_encoded_label = np.array(k_hot_encoded_label).astype(int)

    return k_hot_encoded_label


def get_class_weight(samples):
    """
    Gets the class weighting for use during training.

    :param samples: List of sample filenames
    :return: Dict of class weights
    """
    emotion_running_counts = np.zeros(7, dtype=int)

    # Count how many times an emotion is present in the samples
    for sample in samples:
        label = read_label(sample)
        emotion_running_counts += label

    # Convert the running count list into a total count dictionary
    emotion_total_counts = {}
    for index, counts in enumerate(emotion_running_counts):
        emotion_total_counts[index] = counts

    # Calculate how many times a class if off from the ideal balance. Ideally,
    # the distribution is uniform so the number of samples per class is equal to
    # the number of samples divided by the number of classes.
    class_weights = {}
    total = sum(emotion_total_counts.values())
    ideal_num_samples = total / emc.NUM_EMOTIONS

    for emotion, count in emotion_total_counts.items():
        class_weights[emotion] = ideal_num_samples / count

    return class_weights


def main():
    samples = get_sample_filenames()
    # print(len(samples))
    get_class_weight(samples)


if __name__ == "__main__":
    main()
