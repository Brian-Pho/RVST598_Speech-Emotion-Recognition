"""
This file deals with processing and loading the CREMA-D database. There are two
options to use this database:
    - Load the database into RAM as a numpy array
    - Process the database into .npy files of log-mel spectrogram

Use the first option if you want quick access to the database or if you need to
apply operations to the entire database such as statistics.

Use the second option if you want to use your disk space instead of RAM and if
you plan on using a data generator to feed a neural network the samples.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from db_common import generate_db_stats, is_outlier, get_label
from src import em_constants as emc
from src.audio_processor.wav import load_wav, process_wav

CRE_MIN_LEN, CRE_MAX_LEN = 60861, 193794
CRE_EMO_INDEX = 2  # The index into the filename for the emotion label
CRE_SR = 16000  # The sampling rate for all Crema-d audio samples
CRE_EMOTION_MAP = {
    "ANG": emc.ANG,
    "DIS": emc.DIS,
    "FEA": emc.FEA,
    "HAP": emc.HAP,
    "NEU": emc.NEU,
    "SAD": emc.SAD,
}
MEL_SPEC_FILENAME = "C_{id}_{emo_label}.npy"


def load_data():
    """
    Loads the CREMA-D database from the cached files. If they don't exist,
    then read the database and create the cache.

    :return: Tuple of (samples, labels) or None if unsuccessful.
    """
    cremad_samples = None
    cremad_labels = None

    try:
        # Attempt to read the cache
        cremad_samples = np.load(dbc.CRE_SAMPLES_CACHE_PATH, allow_pickle=True)
        cremad_labels = np.load(dbc.CRE_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the CREMA-D cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        cremad_samples, cremad_labels = read_data()
        np.save(dbc.CRE_SAMPLES_CACHE_PATH, cremad_samples, allow_pickle=True)
        np.save(dbc.CRE_LABELS_CACHE_PATH, cremad_labels, allow_pickle=True)
        print("Successfully cached the CREMA-D database.")

    finally:
        # Pack the data
        cremad_data = (cremad_samples, cremad_labels)
        return cremad_data


def read_data():
    """
    Reads the CREMA-D database into tensors.

    Sample output:
        (array([-0.00228882, -0.00204468, -0.00180054, ...,  0.        ,
        0.        ,  0.        ], dtype=float32), 2)

    :return: Tuple of (samples, labels) where the samples are a tensor of
             varying shape due to varying audio lengths, and the labels are an
             array of integers. The shape is (1440,) from 1440 audio files.
    """
    samples = []
    labels = []
    wav_folder = os.path.join(dbc.CRE_DB_PATH, "AudioWAV")

    for sample_filename in os.listdir(wav_folder):
        print("Processing file:", sample_filename)
        sample_path = os.path.join(wav_folder, sample_filename)

        # Read the sample
        samples.append(load_wav(sample_path))

        # Read the label
        labels.append(
            get_label(sample_filename, "_", CRE_EMO_INDEX, CRE_EMOTION_MAP))

    return np.array(samples), np.array(labels)


def read_to_melspecgram():
    """
    Reads the raw waveforms and converts them into log-mel spectrograms which
    are stored. This is an alternative to read_data() and load_data() to prevent
    using too much ram. Trades RAM for disk space.
    """
    id_counter = 0

    wav_folder = os.path.join(dbc.CRE_DB_PATH, "AudioWAV")

    for sample_filename in os.listdir(wav_folder):
        sample_path = os.path.join(wav_folder, sample_filename)

        # Read the sample
        wav = load_wav(sample_path)

        # Check if it's an outlier
        if is_outlier(wav, lower=CRE_MIN_LEN, upper=CRE_MAX_LEN):
            continue

        # Process the sample into a log-mel spectrogram
        melspecgram = process_wav(wav)

        plt.pcolormesh(melspecgram, cmap="magma")
        plt.show()

        # Read the label
        label = get_label(sample_filename, "_", CRE_EMO_INDEX, CRE_EMOTION_MAP)

        # Save the log-mel spectrogram to use later
        mel_spec_path = os.path.join(
            dbc.PROCESS_DB_PATH, MEL_SPEC_FILENAME.format(
                id=id_counter, emo_label=label))
        np.save(mel_spec_path, melspecgram, allow_pickle=True)
        id_counter += 1


def main():
    """
    Local testing and cache creation.
    """
    cremad_samples, cremad_labels = load_data()
    # print(cremad_samples.shape)
    # print(cremad_labels.shape)
    generate_db_stats(cremad_samples, cremad_labels)
    # read_to_melspecgram()


if __name__ == "__main__":
    main()
