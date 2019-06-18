import os

import librosa
import numpy as np
from db_constants import (
    RAVDESS_DB_PATH, RAVDESS_SAMPLES_CACHE_PATH, RAVDESS_LABELS_CACHE_PATH)
from src import constants as c

NUM_ACTORS = 24
RAVDESS_EMOTION_MAP = {
    "01": c.NEU,
    "02": c.NEU,  # Map calm to neutral
    "03": c.HAP,
    "04": c.SAD,
    "05": c.ANG,
    "06": c.FEA,
    "07": c.DIS,
    "08": c.SUR,
}


def load_data():
    """
    Loads the RAVDESS database from the cached files. If they don't exist,
    then read the database and create the cache.

    :return: Tuple of (samples, labels) or None if unsuccessful.
    """
    ravdess_samples = None
    ravdess_labels = None

    try:
        # Attempt to read the cache
        ravdess_samples = np.load(RAVDESS_SAMPLES_CACHE_PATH, allow_pickle=True)
        ravdess_labels = np.load(RAVDESS_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the RAVDESS cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        ravdess_samples, ravdess_labels = read_data()
        np.save(RAVDESS_SAMPLES_CACHE_PATH, ravdess_samples, allow_pickle=True)
        np.save(RAVDESS_LABELS_CACHE_PATH, ravdess_labels, allow_pickle=True)
        print("Successfully cached the RAVDESS database.")

    finally:
        ravdess_data = (ravdess_samples, ravdess_labels)
        return ravdess_data


def read_data():
    """
    Reads the RAVDESS database into tensors.

    Sample output:
        (array([[array([ 1.5591205e-07, -1.5845627e-07,  1.5362870e-07, ...,
        0.0000000e+00,  0.0000000e+00,  0.0000000e+00], dtype=float32),
        22050]], dtype=object), array([0, 0, 0, ..., 6]))

    :return: Tuple of (samples, labels) where the samples are a tensor of shape
             (files, data), and the labels are an array of integers. The data is
             formatted as (audio time series, sampling rate).
    """
    samples = []
    labels = []
    # for actor in range(1, NUM_ACTORS + 1):
    for actor in range(1, 2):
        actor_foldername = "Actor_{:02d}".format(actor)
        actor_path = os.path.join(RAVDESS_DB_PATH, actor_foldername)

        for sample_filename in os.listdir(actor_path):
            sample_path = os.path.join(actor_path, sample_filename)

            # Read the sample
            # print(sample_filename)
            samples.append(librosa.load(sample_path))

            # Read the label
            labels.append(_interpret_label(sample_filename))

    return np.array(samples), np.array(labels)


def _interpret_label(filename):
    """
    Given a filename, it returns an integer representing the emotion label of
    the file/sample.
    :return: Integer
    """
    # Parse emotion ID from filename. It's the third number from the left
    # according to https://zenodo.org/record/1188976.
    emotion_id = filename.split("-")[2]
    emotion = RAVDESS_EMOTION_MAP[emotion_id]

    # Return a new emotion ID that's standardized across databases.
    return c.EMOTION_MAP[emotion]


def main():
    ravdess_samples, ravdess_labels = load_data()
    print(ravdess_samples.shape)
    # print(ravdess_samples[0][0])  # Amplitude data
    # print(ravdess_samples[0][1])  # Sampling rate data
    print(ravdess_labels.shape)
    # print(ravdess_labels[0])
    # print(ravdess_data)
    # print(ravdess_data[0][1])
    # for data in ravdess_data:
    #     print(data[1])
    # print(load_data())


if __name__ == "__main__":
    main()
