"""
This file deals with processing and loading the IEMOCAP database. There are two
options to use this database:
    - Load the database into RAM as a numpy array
    - Process the database into .npy files of log-mel spectrogram

Use the first option if you want quick access to the database or if you need to
apply operations to the entire database such as statistics.

Use the second option if you want to use your disk space instead of RAM and if
you plan on using a data generator to feed a neural network the samples.
"""

import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from common import load_wav, calculate_bounds, process_wav, is_outlier
from src import constants as c


def generate_stats():
    pass


def load_data():
    """
    Loads the IEMOCAP database from the cached files. If they don't exist,
    then read the database and create the cache.

    :return: Tuple of (samples, labels) or None if unsuccessful.
    """
    iemocap_samples = None
    iemocap_labels = None

    try:
        # Attempt to read the cache
        iemocap_samples = np.load(dbc.IEM_SAMPLES_CACHE_PATH, allow_pickle=True)
        iemocap_labels = np.load(dbc.IEM_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the IEMOCAP cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        iemocap_samples, iemocap_labels = read_data()
        np.save(dbc.IEM_SAMPLES_CACHE_PATH, iemocap_samples, allow_pickle=True)
        np.save(dbc.IEM_LABELS_CACHE_PATH, iemocap_labels, allow_pickle=True)
        print("Successfully cached the IEMOCAP database.")

    finally:
        # Pack the data
        iemocap_data = (iemocap_samples, iemocap_labels)
        return iemocap_data


def read_data():
    """
    Reads the RAVDESS database into tensors.

    Sample output:
        (array([ 3.0517578e-05,  3.0517578e-05,  3.0517578e-05, ...,
        0.0000000e+00, -3.0517578e-05,  0.0000000e+00], dtype=float32), 0)

    :return: Tuple of (samples, labels) where the samples are a tensor of
             varying shape due to varying audio lengths, and the labels are an
             array of integers. The shape is (1440,) from 1440 audio files.
    """
    samples = []
    labels = []
    data_path = os.path.join(dbc.IEM_DB_PATH, "data")
    labels_path = os.path.join(dbc.IEM_DB_PATH, "labels")

    # for num_sess in range(1, 6):
    for num_sess in range(1, 2):
        sess_foldername = "S{}".format(num_sess)
        print("Processing session:", sess_foldername)

        sess_data_path = os.path.join(data_path, sess_foldername)

        for perform in os.listdir(sess_data_path):
            perform_path = os.path.join(sess_data_path, perform)
            print("Processing performance:", perform)

            for sample_filename in os.listdir(perform_path):
                sample_path = os.path.join(perform_path, sample_filename)

                wav = load_wav(sample_path)
                melspecgram = process_wav(wav)
                # melspecgram = librosa.feature.melspectrogram(wav)
                plt.pcolormesh(melspecgram)
                plt.colorbar()
                plt.show()

        sess_label_path = os.path.join(labels_path, sess_foldername)

        for perform in os.listdir(sess_label_path):
            perform_path = os.path.join(sess_label_path, perform)


def read_to_melspecgram():
    pass


def _interpret_label(filename):
    pass


def main():
    read_data()


if __name__ == "__main__":
    main()
