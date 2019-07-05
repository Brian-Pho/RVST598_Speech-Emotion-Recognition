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

import numpy as np

import db_constants as dbc
from common import load_wav, process_wav, generate_db_stats
from src import constants as c

IEM_MIN_LEN, IEM_MAX_LEN = None, None
IEM_SR = 16000  # The sampling rate for all Ravdess audio samples
IEM_EMOTION_MAP = {
    "Neutral state": c.NEU,
    "Happiness": c.HAP,
    "Sadness": c.SAD,
    "Anger": c.ANG,
    "Fear": c.FEA,
    "Disgust": c.DIS,
    "Surprise": c.SUR,
}
MEL_SPEC_FILENAME = "I_{id}_{emo_label}.npy"


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
        print(iemocap_samples.shape, iemocap_labels.shape)
        np.save(dbc.IEM_SAMPLES_CACHE_PATH, iemocap_samples, allow_pickle=True)
        np.save(dbc.IEM_LABELS_CACHE_PATH, iemocap_labels, allow_pickle=True)
        print("Successfully cached the IEMOCAP database.")

    finally:
        # Pack the data
        iemocap_data = (iemocap_samples, iemocap_labels)
        return iemocap_data


def read_data():
    """
    Reads the IEMOCAP database into tensors.

    Sample output:
        (array([ 3.0517578e-05,  3.0517578e-05,  3.0517578e-05, ...,
        0.0000000e+00, -3.0517578e-05,  0.0000000e+00], dtype=float32), 0)

    :return: Tuple of (samples, labels) where the samples are a tensor of
             varying shape due to varying audio lengths, and the labels are an
             array of integers.
    """
    samples = []
    labels = []
    data_path = os.path.join(dbc.IEM_DB_PATH, "data")
    labels_path = os.path.join(dbc.IEM_DB_PATH, "labels")

    for num_sess in range(1, 6):
        sess_foldername = "S{}".format(num_sess)
        print("Processing session:", sess_foldername)

        sess_data_path = os.path.join(data_path, sess_foldername)

        for perform in os.listdir(sess_data_path):
            perform_path = os.path.join(sess_data_path, perform)
            print("Processing performance:", perform)

            for sample_filename in os.listdir(perform_path):
                sample_path = os.path.join(perform_path, sample_filename)

                wav = load_wav(sample_path)
                samples.append(wav)

                # melspecgram = process_wav(wav)
                # melspecgram = librosa.feature.melspectrogram(wav)
                # plt.pcolormesh(melspecgram)
                # plt.colorbar()
                # plt.show()

        # sess_label_path = os.path.join(labels_path, sess_foldername)
        #
        # for perform in os.listdir(sess_label_path):
        #     perform_path = os.path.join(sess_label_path, perform)

    return np.array(samples), np.array(labels)


def read_to_melspecgram():
    """
    Reads the raw waveforms and converts them into log-mel spectrograms which
    are stored. This is an alternative to read_data() and load_data() to prevent
    using too much ram. Trades RAM for disk space.
    """
    id_counter = 0
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

                # Read the sample
                wav = load_wav(sample_path)

                # Process the sample into a log-mel spectrogram
                melspecgram = process_wav(wav)

                # plt.pcolormesh(melspecgram)
                # plt.colorbar()
                # plt.show()

                # Save the log-mel spectrogram to use later
                mel_spec_path = os.path.join(
                    dbc.PROCESS_DB_PATH, MEL_SPEC_FILENAME.format(
                        id=id_counter, emo_label=label))
                np.save(mel_spec_path, melspecgram, allow_pickle=True)
                id_counter += 1

        sess_label_path = os.path.join(labels_path, sess_foldername)

        for perform in os.listdir(sess_label_path):
            perform_path = os.path.join(sess_label_path, perform)


def _interpret_label(filename):
    # If the label is not mappable, then send False and drop that sample
    pass


def main():
    iemocap_samples, iemocap_labels = load_data()
    generate_db_stats(iemocap_samples, iemocap_samples)


if __name__ == "__main__":
    main()
