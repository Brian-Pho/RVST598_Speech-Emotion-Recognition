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
from db_common import generate_db_stats
from src import em_constants as emc
from src.audio_processor.wav import load_wav, process_wav

IEM_MIN_LEN, IEM_MAX_LEN = None, None
IEM_SR = 16000  # The sampling rate for all Ravdess audio samples
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
        sess_labels_path = os.path.join(labels_path, sess_foldername)

        for perform in os.listdir(sess_data_path):
            perform_data_path = os.path.join(sess_data_path, perform)
            perform_labels_path = os.path.join(
                sess_labels_path, perform + ".txt")
            print("Processing performance:", perform)

            perform_label_map = get_label_map(perform_labels_path)
            # print(perform_label_map)

            for sample_filename in os.listdir(perform_data_path):
                # # If the label is empty, then drop the sample
                # label = perform_label_map[sample_filename]
                # if not label:
                #     continue

                # Read the sample
                sample_path = os.path.join(perform_data_path, sample_filename)
                samples.append(load_wav(sample_path))

                # Remove the file extension and get and read the label
                sample_filename = os.path.splitext(sample_filename)[0]
                labels.append(perform_label_map[sample_filename])

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

    for num_sess in range(1, 6):
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
                melspecgram = process_wav(wav)

                # plt.pcolormesh(melspecgram)
                # plt.colorbar()
                # plt.show()

                # Get the label
                sample_filename = os.path.splitext(sample_filename)[0]
                label = "_".join(
                    str(emo) for emo in perform_label_map[sample_filename])

                # Save the log-mel spectrogram to use later
                mel_spec_path = os.path.join(
                    dbc.PROCESS_DB_PATH, MEL_SPEC_FILENAME.format(
                        id=id_counter, emo_label=label))
                np.save(mel_spec_path, melspecgram, allow_pickle=True)
                id_counter += 1


def get_label_map(labels_path):
    """
    Gets the label map for every sample in a performance.

    Sample output:
        {
            'Ses01F_impro01_F000':
                array([1., 0., 0., 0., 0., 0., 0.]),
            'Ses01F_impro01_F001':
                array([1., 0., 0., 0., 0., 0., 0.]), ...
        }

    :param labels_path: Path to the labels txt
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
                sample_emotions = _interpret_label(sample_emotions)
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
            emotions.remove("")

            sample_emotions += emotions

    return sample_emo_map


def _interpret_label(labels):
    """
    Interprets the emotion(s) from the list of labels to the standard format.

    :param labels: the list of emotions for a sample
    :return: List
    """
    # Not considering the "Other" emotion labels
    labels = [label for label in labels if label != "Other"]
    # Convert the emotions into the standard emotion numbers in constants
    standard_emotions = [
        emc.EMOTION_MAP[IEM_EMOTION_MAP[label]] for label in labels]

    # Convert the emotion numbers into an array where the index is the emotion
    # and the value is the number of votes for that emotion
    unique, counts = np.unique(standard_emotions, return_counts=True)
    one_hot_emotions = np.zeros(emc.NUM_EMOTIONS)
    for emo_index, emo_count in zip(unique, counts):
        one_hot_emotions[emo_index] = emo_count

    # Only count the emotions with the highest amount of votes
    scaled_emotions = one_hot_emotions / np.max(one_hot_emotions)
    most_voted_emotions = np.floor(scaled_emotions).astype(int)

    # If they're all zero, then this sample doesn't fit with the set of emotion
    # that we're considering so drop it
    if not np.any(most_voted_emotions):
        print("Sample unused")
        return False

    return most_voted_emotions


def main():
    iemocap_samples, iemocap_labels = load_data()
    generate_db_stats(iemocap_samples, iemocap_samples)
    read_to_melspecgram()


if __name__ == "__main__":
    main()
