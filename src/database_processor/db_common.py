"""
This file holds common functions across all database processing such as
calculating statistics.
"""

import numpy as np

from src import em_constants as emc


def is_outlier(wav, lower, upper):
    """
    Checks if an audio sample is an outlier. Bounds are inclusive.

    :param wav: The audio time series data points
    :param lower: The lower bound
    :param upper: The upper bound
    :return: Boolean
    """
    return False if lower <= len(wav) <= upper else True


def get_label(filename, delimiter, index, db_emo_map):
    """
    Gets the label from a sample's filename.

    :param filename: The sample's filename
    :param delimiter: The delimiter used in the filename
    :param index: Where in the filename the label/emotion is located
    :param db_emo_map: The database-specific emotion mapping
    :return: The label k-hot encoded to this program's standard emotion map or
             False if the label doesn't map to the standard emotions
    """
    label = filename.split(delimiter)[index]
    standard_emotion = db_emo_map[label]
    emotion_id = [emc.EMOTION_MAP[standard_emotion]]
    return k_hot_encode_label(emotion_id)


def repr_label(label):
    """
    Represents a label in a filename-friendly format. Mostly used for the
    "read_to_melspecgram()" function to write out labels in the filename.

    Sample input:
        [1. 0. 0. 0. 0. 0. 0.]

    Sample output:
        "1_0_0_0_0_0_0"

    :param label: Numpy array representing the k-hot encoded label
    :return: String representation of the label
    """
    return "_".join(str(emo) for emo in label)


def k_hot_encode_label(label):
    """
    K-hot encodes a label. Takes a list of emotion IDs and returns a list
    encoding the most voted for emotion.

    Sample input:
        [0, 1, 2, 0, 6, 2]

    Sample output:
        [1, 0, 1, 0, 0, 0, 0]

    :param label: List of labels to encode
    :return: List of k-hot encoded labels or False if the label is unused
    """
    #  If there's only one label/vote, then use the quicker method of encoding
    if len(label) == 1:
        return _one_hot_encode_label(label)

    # Convert the emotion numbers into an array where the index is the emotion
    # and the value is the number of votes for that emotion
    unique, counts = np.unique(label, return_counts=True)
    k_hot_label = np.zeros(emc.NUM_EMOTIONS)
    for emo_index, emo_count in zip(unique, counts):
        k_hot_label[emo_index] = emo_count

    # Only count the emotions with the highest amount of votes
    k_hot_label = k_hot_label / np.max(k_hot_label)
    k_hot_label = np.floor(k_hot_label).astype(int)

    # If they're all zero, then this sample doesn't fit with the set of labels
    # that we're considering so drop it
    if not np.any(k_hot_label):
        print("No usable label.")
        return False

    return k_hot_label


def _one_hot_encode_label(label):
    """
    One hot encodes a label. Private function to quickly one-hot encode a label.

    Sample input:
        [4]

    Sample output:
        [0, 0, 0, 0, 1, 0, 0]

    :param label: A list with one label (length is one)
    :return: One-hot encoding of the label
    """
    one_hot_label = np.zeros(emc.NUM_EMOTIONS, dtype=int)
    one_hot_label[label[0]] = 1
    return one_hot_label


def inverse_k_hot_encode_label(k_hot_label):
    """
    Inverse a k-hot encoded label back to emotion ids.

    Sample input:
        [1, 0, 0, 0, 1, 0, 0]

    Sample output:
        [0, 4]

    :param k_hot_label: A list of the k-hot encoded label
    :return: A list of the emotion ids in the label
    """
    return np.where(k_hot_label == 1)[0]
