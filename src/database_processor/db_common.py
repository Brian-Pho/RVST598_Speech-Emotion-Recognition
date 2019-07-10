"""
This file holds common functions across all database processing such as
calculating statistics.
"""

import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from src import em_constants as emc


def get_label(filename, delimiter, index, db_emo_map):
    """
    Gets the label from a sample's filename.

    :param filename: The sample's filename
    :param delimiter: The delimiter used in the filename
    :param index: Where in the filename the label/emotion is located
    :param db_emo_map: The database-specific emotion mapping
    :return: The label k-hot encoded to this program's standard emotion map
    """
    label = filename.split(delimiter)[index]
    standard_emotion = db_emo_map[label]
    emotion_id = emc.EMOTION_MAP[standard_emotion]
    return k_hot_encode_label(list(emotion_id))


def calculate_bounds(data, num_std):
    """
    Calculates the lower and upper bound given a distribution and standard
    deviation.

    :param data: The dataset/distribution
    :param num_std: The number of standard deviations to set the bounds
    :return: Tuple, of the lower and upper bound
    """
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * num_std
    lower, upper = data_mean - cut_off, data_mean + cut_off

    return lower, upper


def is_outlier(wav, lower, upper):
    """
    Checks if an audio sample is an outlier. Bounds are inclusive.

    :param wav: The audio time series data points
    :param lower: The lower bound
    :param upper: The upper bound
    :return: Boolean
    """
    return False if lower <= len(wav) <= upper else True


def generate_db_stats(samples, labels):
    """
    Generates statistics from the given samples and labels.

    :param samples: Samples from the database
    :param labels: Labels from the database
    """
    # Calculate the emotion class percentages. The neutral class has the most
    # samples due to combining it with the calm class.
    emo_labels = [emc.INVERT_EMOTION_MAP[label] for label in labels]
    unique, counts = np.unique(emo_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    plt.pie(x=counts, labels=unique)
    plt.show()

    # Calculate the distribution of tensor shapes for the samples
    audio_lengths = [len(sample) for sample in samples]
    print("Shortest:", min(audio_lengths), "Longest:", max(audio_lengths))

    lower, upper = calculate_bounds(audio_lengths, dbc.NUM_STD_CUTOFF)
    print("Lower bound:", lower, "Upper bound:", upper)

    num_outliers = [length for length in audio_lengths
                    if length < lower or length > upper]
    print("Num outliers:", len(num_outliers))

    audio_cropped_lengths = [length for length in audio_lengths
                             if lower <= length <= upper]
    print("Num included:", len(audio_cropped_lengths))

    unique, counts = np.unique(audio_cropped_lengths, return_counts=True)
    data_min = unique[0]
    data_max = unique[-1]
    print(samples.shape, data_min, data_max)

    plt.bar(unique, counts, width=700)
    plt.xlabel("Number of Data Points")
    plt.ylabel("Number of Samples")
    plt.title("The Distribution of Samples with Number of Data Points")
    plt.show()


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
    one_hot_label = np.zeros(emc.NUM_EMOTIONS)
    one_hot_label[label[0]] = 1
    return one_hot_label
