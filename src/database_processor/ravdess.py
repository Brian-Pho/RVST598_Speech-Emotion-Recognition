"""
This file deals with processing and loading the RAVDESS database. There are two
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
from db_common import get_label, repr_label
from spectrogram import display_melspecgram
from src import em_constants as emc
from src.audio_processor.wav import load_wav, process_wav, remove_first_last_sec

NUM_ACTORS = 24
RAV_EMO_INDEX = 2  # The index into the filename for the emotion label
RAV_SR = 48000  # The sampling rate for all Ravdess audio samples
RAV_EMOTION_MAP = {
    "01": emc.NEU,
    "02": emc.NEU,  # Map calm to neutral
    "03": emc.HAP,
    "04": emc.SAD,
    "05": emc.ANG,
    "06": emc.FEA,
    "07": emc.DIS,
    "08": emc.SUR,
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
        ravdess_samples = np.load(dbc.RAV_SAMPLES_CACHE_PATH, allow_pickle=True)
        ravdess_labels = np.load(dbc.RAV_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the RAVDESS cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        ravdess_samples, ravdess_labels = read_data()
        np.save(dbc.RAV_SAMPLES_CACHE_PATH, ravdess_samples, allow_pickle=True)
        np.save(dbc.RAV_LABELS_CACHE_PATH, ravdess_labels, allow_pickle=True)
        print("Successfully cached the RAVDESS database.")

    finally:
        # Pack the data
        ravdess_data = (ravdess_samples, ravdess_labels)
        return ravdess_data


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

    for actor in range(1, NUM_ACTORS + 1):
        actor_foldername = "Actor_{:02d}".format(actor)
        actor_path = os.path.join(dbc.RAV_DB_PATH, actor_foldername)
        print("Processing actor:", actor_foldername)

        for sample_filename in os.listdir(actor_path):
            # Read the sample and remove the first and last second
            sample_path = os.path.join(actor_path, sample_filename)
            wav = load_wav(sample_path)
            samples.append(remove_first_last_sec(wav, RAV_SR))

            # Read the label
            label = get_label(
                sample_filename, "-", RAV_EMO_INDEX, RAV_EMOTION_MAP)
            labels.append(label)

    return np.array(samples), np.array(labels)


def read_to_melspecgram():
    """
    Reads the raw waveforms and converts them into log-mel spectrograms which
    are stored. This is an alternative to read_data() and load_data() to prevent
    using too much ram. Trades RAM for disk space.
    """
    id_counter = 0

    for actor in range(1, NUM_ACTORS + 1):
        actor_foldername = "Actor_{:02d}".format(actor)
        actor_path = os.path.join(dbc.RAV_DB_PATH, actor_foldername)
        print("Processing actor:", actor_foldername)

        for sample_filename in os.listdir(actor_path):
            # Read the sample and remove the first and last second
            sample_path = os.path.join(actor_path, sample_filename)
            wav = remove_first_last_sec(load_wav(sample_path), RAV_SR)

            # Process the sample into a log-mel spectrogram
            melspecgram = process_wav(wav)

            # Display the spectrogram
            display_melspecgram(melspecgram)

            # Read the label
            label = get_label(
                sample_filename, "-", RAV_EMO_INDEX, RAV_EMOTION_MAP)
            label = repr_label(label)

            # Save the log-mel spectrogram to use later
            mel_spec_path = os.path.join(
                dbc.PROCESS_DB_PATH, dbc.RAV_MEL_SPEC_FN.format(
                    id=id_counter, emo_label=label))
            np.save(mel_spec_path, melspecgram)
            id_counter += 1


def main():
    """
    Local testing and cache creation.
    """
    # ravdess_samples, ravdess_labels = load_data()
    # print(ravdess_samples.shape)
    # print(ravdess_labels.shape)
    read_to_melspecgram()


if __name__ == "__main__":
    main()
