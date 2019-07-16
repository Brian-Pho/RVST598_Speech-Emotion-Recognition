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

import csv
import os

import numpy as np

import db_constants as dbc
from db_common import k_hot_encode_label, repr_label
from spectrogram import display_melspecgram
from src import em_constants as emc
from src.audio_processor.wav import load_wav, process_wav

CRE_EMO_INDEX = 2  # The index into the filename for the emotion label
CRE_SR = 16000  # The sampling rate for all Crema-d audio samples
CRE_EMOTION_MAP = {
    "A": emc.ANG,
    "D": emc.DIS,
    "F": emc.FEA,
    "H": emc.HAP,
    "N": emc.NEU,
    "S": emc.SAD,
}


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
        cremad_samples = np.load(dbc.CRE_SAMPLES_CACHE_PATH)
        cremad_labels = np.load(dbc.CRE_LABELS_CACHE_PATH)
        print("Successfully loaded the CREMA-D cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        cremad_samples, cremad_labels = read_data()
        np.save(dbc.CRE_SAMPLES_CACHE_PATH, cremad_samples)
        np.save(dbc.CRE_LABELS_CACHE_PATH, cremad_labels)
        print("Successfully cached the CREMA-D database.")

    finally:
        # Pack the data
        cremad_data = (cremad_samples, cremad_labels)
        return cremad_data


def read_data():
    """
    Reads the CREMA-D database into np.array.

    Sample output:
        (array([-0.00228882, -0.00204468, -0.00180054, ...,  0.        ,
        0.        ,  0.        ], dtype=float32), 2)

    :return: Tuple of (samples, labels) where the samples are a np.array and the
    labels are an array of integers.
    """
    samples = []
    labels = []
    wav_folder = os.path.join(dbc.CRE_DB_PATH, "AudioWAV")
    labels_path = os.path.join(dbc.CRE_DB_PATH, "tabulatedVotes.csv")
    label_map = get_label_map(labels_path)

    for sample_filename in os.listdir(wav_folder):
        print("Processing file:", sample_filename)

        # Read the sample
        sample_path = os.path.join(wav_folder, sample_filename)
        samples.append(load_wav(sample_path))

        # Read the label
        sample_name = os.path.splitext(sample_filename)[0]
        label = label_map[sample_name]
        labels.append(label)

    return np.array(samples), np.array(labels)


def read_to_melspecgram():
    """
    Reads the raw waveforms and converts them into log-mel spectrograms which
    are stored. This is an alternative to read_data() and load_data() to prevent
    using too much ram. Trades RAM for disk space.
    """
    id_counter = 0

    wav_folder = os.path.join(dbc.CRE_DB_PATH, "AudioWAV")
    labels_path = os.path.join(dbc.CRE_DB_PATH, "tabulatedVotes.csv")
    label_map = get_label_map(labels_path)

    for sample_filename in os.listdir(wav_folder):
        # Read the sample
        sample_path = os.path.join(wav_folder, sample_filename)
        wav = load_wav(sample_path)

        # Process the sample into a log-mel spectrogram
        melspecgram = process_wav(wav, noisy=True)

        # Display the spectrogram
        display_melspecgram(melspecgram)

        # Read the label
        sample_name = os.path.splitext(sample_filename)[0]
        label = label_map[sample_name]
        label = repr_label(label)

        # Save the log-mel spectrogram to use later
        mel_spec_filename = dbc.CRE_MEL_SPEC_FN.format(
            id=id_counter, emo_label=label)
        mel_spec_path = os.path.join(dbc.PROCESS_DB_PATH, mel_spec_filename)
        np.save(mel_spec_path, melspecgram)
        id_counter += 1


def get_label_map(labels_path, encode=True):
    """
    Gets the label map for every sample in the CREMA-D database.

    :param labels_path: Path to tabulatedVotes.csv
    :param encode: Bool to k-hot encode the label or not
    :return: Dict
    """
    sample_emo_map = {}

    file_name_header = "fileName"
    emo_vote_header = "emoVote"

    with open(labels_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            sample_name = row[file_name_header]
            sample_emotions = row[emo_vote_header].split(":")

            if sample_name not in sample_emo_map:
                sample_emo_map[sample_name] = sample_emotions
            else:
                sample_emo_map[sample_name] += sample_emotions

    # K-hot encode the emotions
    if encode:
        for sample, emo in sample_emo_map.items():
            sample_emo_map[sample] = encode_label(emo)

    return sample_emo_map


def encode_label(labels):
    """
    Encodes a label into a k-hot encoded label.

    :param labels: the list of emotions for a sample
    :return: List
    """
    # Not considering emotions outside of our defined list
    labels = [label for label in labels if label in CRE_EMOTION_MAP.keys()]
    # Convert the emotions into the standard emotion ids
    standard_emo_ids = [
        emc.EMOTION_MAP[CRE_EMOTION_MAP[label]] for label in labels]

    return k_hot_encode_label(standard_emo_ids)


def main():
    """
    Local testing and cache creation.
    """
    # cremad_samples, cremad_labels = load_data()
    # print(cremad_samples.shape)
    # print(cremad_labels.shape)
    read_to_melspecgram()


if __name__ == "__main__":
    main()
