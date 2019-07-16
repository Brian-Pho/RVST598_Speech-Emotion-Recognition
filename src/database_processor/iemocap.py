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
from db_common import k_hot_encode_label, repr_label
from spectrogram import display_melspecgram
from src import em_constants as emc
from src.audio_processor.wav import load_wav, process_wav

NUM_SESS = 5
IEM_SR = 16000  # The sampling rate for all Iemocap audio samples
IEM_EMOTION_MAP = {
    "Neutral": emc.NEU,
    "Happiness": emc.HAP,
    "Excited": emc.HAP,  # Map excited to happy
    "Sadness": emc.SAD,
    "Anger": emc.ANG,
    "Frustration": emc.ANG,  # Map frustration to anger
    "Fear": emc.FEA,
    "Disgust": emc.DIS,
    "Surprise": emc.SUR,
}


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
        iemocap_samples = np.load(dbc.IEM_SAMPLES_CACHE_PATH)
        iemocap_labels = np.load(dbc.IEM_LABELS_CACHE_PATH)
        print("Successfully loaded the IEMOCAP cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        iemocap_samples, iemocap_labels = read_data()
        np.save(dbc.IEM_SAMPLES_CACHE_PATH, iemocap_samples)
        np.save(dbc.IEM_LABELS_CACHE_PATH, iemocap_labels)
        print("Successfully cached the IEMOCAP database.")

    finally:
        # Pack the data
        iemocap_data = (iemocap_samples, iemocap_labels)
        return iemocap_data


def read_data():
    """
    Reads the IEMOCAP database into np.array.

    Sample output:
        (array([ 3.0517578e-05,  3.0517578e-05,  3.0517578e-05, ...,
        0.0000000e+00, -3.0517578e-05,  0.0000000e+00], dtype=float32), 0)

    :return: Tuple of (samples, labels) where the samples are a np.array and the
    labels are an array of integers.
    """
    samples = []
    labels = []
    data_path = os.path.join(dbc.IEM_DB_PATH, "data")
    labels_path = os.path.join(dbc.IEM_DB_PATH, "labels")

    for num_sess in range(1, NUM_SESS + 1):
        sess_foldername = "S{}".format(num_sess)
        print("Processing session:", sess_foldername)

        sess_data_path = os.path.join(data_path, sess_foldername)
        sess_labels_path = os.path.join(labels_path, sess_foldername)

        for perform in os.listdir(sess_data_path):
            perform_data_path = os.path.join(sess_data_path, perform)
            perform_labels_path = os.path.join(
                sess_labels_path, perform + ".txt")
            print("Processing performance:", perform)

            perform_label_map = get_label_map(perform_labels_path)

            for sample_filename in os.listdir(perform_data_path):
                # Read the label and if it's empty, then drop the sample
                sample_name = os.path.splitext(sample_filename)[0]
                label = perform_label_map[sample_name]
                if not label:
                    print("Not using sample:", sample_filename)
                    continue

                # Read the sample
                sample_path = os.path.join(perform_data_path, sample_filename)
                samples.append(load_wav(sample_path))

                labels.append(label)

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

    for num_sess in range(1, NUM_SESS + 1):
        sess_foldername = "S{}".format(num_sess)
        print("Processing session:", sess_foldername)

        sess_data_path = os.path.join(data_path, sess_foldername)
        sess_labels_path = os.path.join(labels_path, sess_foldername)

        for perform in os.listdir(sess_data_path):
            perform_data_path = os.path.join(sess_data_path, perform)
            perform_labels_path = os.path.join(
                sess_labels_path, perform + ".txt")
            print("Processing performance:", perform)

            perform_label_map = get_label_map(perform_labels_path)

            for sample_filename in os.listdir(perform_data_path):
                # Read the sample
                sample_path = os.path.join(perform_data_path, sample_filename)
                wav = load_wav(sample_path)

                # Process the sample into a log-mel spectrogram
                melspecgram = process_wav(wav, noisy=True)

                # Display the spectrogram
                display_melspecgram(melspecgram)

                # Get the label
                sample_filename = os.path.splitext(sample_filename)[0]
                label = repr_label(perform_label_map[sample_filename])

                # Save the log-mel spectrogram to use later
                mel_spec_filename = dbc.IEM_MEL_SPEC_FN.format(
                    id=id_counter, emo_label=label)
                mel_spec_path = os.path.join(
                    dbc.PROCESS_DB_PATH, mel_spec_filename)
                np.save(mel_spec_path, melspecgram)
                id_counter += 1


def get_label_map(labels_path, encode=True):
    """
    Gets the label map for every sample in a performance.

    Sample output:
        {
            'Ses01F_impro01_F000':
                array([1., 0., 0., 0., 0., 0., 0.]),
            'Ses01F_impro01_F001':
                array([1., 0., 0., 0., 0., 0., 0.]), ...
        }

    :param labels_path: Path to Ses##M/F_impro##.txt
    :param encode: Bool to k-hot encode the label or not
    :return: Dict
    """
    sample_emo_map = {}

    sample_name = ""
    sample_emotions = []
    parse_emotion_flag = False

    with open(labels_path, "r") as label_file:
        for line in label_file:
            # Starts the parsing of a sample
            if line.startswith("["):
                sample_name = line.split()[3]
                # Reset the emotion list for the next sample
                sample_emotions = []
                parse_emotion_flag = True
                continue

            if not parse_emotion_flag:
                continue

            # Stop parsing when the line starts with "A"
            if line.startswith("A"):
                if encode:
                    sample_emotions = encode_label(sample_emotions)
                sample_emo_map[sample_name] = sample_emotions
                parse_emotion_flag = False
                continue

            # Remove all whitespace from line
            line = ''.join(line.split())

            # The emotions are always between ":" and "(" characters
            line = line.split(":")[1]
            line = line.split("(")[0]
            emotions = line.split(";")
            # Remove the blank string that is created when splitting
            emotions = list(filter(None, emotions))

            sample_emotions += emotions

    return sample_emo_map


def encode_label(labels):
    """
    Encodes a label into a k-hot encoded label.

    :param labels: the list of emotions for a sample
    :return: List
    """
    # Not considering emotions outside of our defined list
    labels = [label for label in labels if label in IEM_EMOTION_MAP.keys()]
    # Convert the emotions into the standard emotion ids
    standard_emo_ids = [
        emc.EMOTION_MAP[IEM_EMOTION_MAP[label]] for label in labels]

    return k_hot_encode_label(standard_emo_ids)


def main():
    # iemocap_samples, iemocap_labels = load_data()
    # print(iemocap_samples.shape)
    # print(iemocap_labels.shape)
    read_to_melspecgram()


if __name__ == "__main__":
    main()
