"""
This file holds functions to generate data for a machine learning model.
"""

import os

import numpy as np
from keras.utils import to_categorical

import nn_constants as nnc
from src import em_constants as emc
from src.database_processor import db_constants as dbc

EMO_INDEX = 2


def batch_generator(sample_fns, batch_size=nnc.BATCH_SIZE):
    """
    Generates a batch for training a neural network. Run indefinitely.

    :param sample_fns: A list of filenames of samples
    :param batch_size: The batch size
    :return: A batch in the form of a tuple (inputs, targets)
    """
    num_samples = len(sample_fns)

    if num_samples < nnc.MIN_NUM_SAMPLES:
        print("Not enough samples to satisfy the threshold for training.",
              "Current number:", num_samples, "Threshold:", nnc.MIN_NUM_SAMPLES)

    batch_start = 0
    # If the number of files is less than a batch, then use all of the files
    batch_end = batch_size if num_samples > batch_size else num_samples

    # Run indefinitely as required by Keras. Will wrap around the samples.
    while True:
        batch_inputs = []
        batch_targets = []

        # Construct a batch
        # print(num_samples, "BS:", batch_start, "BE:", batch_end)
        for sample_fn in sample_fns[batch_start:batch_end]:
            # Load a sample
            sample_path = os.path.join(dbc.PROCESS_DB_PATH, sample_fn)
            sample = np.load(sample_path, allow_pickle=False)

            # Load a label
            label = _interpret_label(sample_fn)

            batch_inputs.append(sample)
            batch_targets.append(label)

        batch_inputs = np.expand_dims(np.array(batch_inputs), axis=3)
        # batch_targets = to_categorical(
        #     batch_targets, num_classes=emc.NUM_EMOTIONS)
        # print(batch_inputs.shape, batch_targets.shape)
        yield (batch_inputs, batch_targets)

        # Shift the start and end for the next batch
        batch_start += batch_size
        batch_end += batch_size

        # If we're at the end but not past it, create the tail-end batch
        if batch_start < num_samples < batch_end:
            batch_end = num_samples

        # If we're past the end, wrap around
        if batch_start >= num_samples:
            batch_start = 0
            batch_end = batch_size if num_samples > batch_size else num_samples


def get_sample_filenames():
    """
    Gets the filenames of the samples in the processed data folder. Doesn't
    append the path to save memory.

    :return: List
    """
    return os.listdir(dbc.PROCESS_DB_PATH)


def _interpret_label(filename):
    """
    Given a filename, it returns an integer representing the emotion label of
    the file/sample.

    :return: Integer
    """
    filename = os.path.splitext(filename)[0]  # Removes the file extension
    return int(filename.split("_")[EMO_INDEX])


def main():
    """
    Local testing.
    """
    samples = get_sample_filenames()
    print(len(samples))

    for inputs, targets in batch_generator(samples[0:34], nnc.BATCH_SIZE):
        print(inputs.shape, targets.shape)


if __name__ == "__main__":
    main()
