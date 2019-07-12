"""
This file holds functions to calculate database statistics such as how many
outliers are outside a certain number of standard deviations and voter
agreement.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from db_common import inverse_k_hot_encode_label
from iemocap import get_label_map, NUM_SESS
from src import em_constants as emc


def calculate_bounds(data, num_std=dbc.NUM_STD_CUTOFF):
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
    emo_labels = []
    for label in labels:
        label = inverse_k_hot_encode_label(label)
        emo_labels = np.concatenate([emo_labels, label])

    unique, counts = np.unique(emo_labels, return_counts=True)
    unique = [emc.INVERT_EMOTION_MAP[label] for label in unique]

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


def calculate_iemocap_agreement():
    """
    Calculates the emotion agreement percentage of the IEMOCAP database.

    Expected output:
        "The IEMOCAP database has an emotion agreement percent of: 63.60%."
    """
    db_agreement = {}
    labels_path = os.path.join(dbc.IEM_DB_PATH, "labels")

    for num_sess in range(1, NUM_SESS + 1):
        sess_foldername = "S{}".format(num_sess)
        print("Processing session:", sess_foldername)

        sess_labels_path = os.path.join(labels_path, sess_foldername)

        for perform in os.listdir(sess_labels_path):
            perform_labels_path = os.path.join(sess_labels_path, perform)

            # Get the agreement percents for each sample in this performance
            perform_label_map = get_label_map(perform_labels_path, encode=False)
            perform_agreement = _calculate_iem_perform_agree(perform_label_map)
            db_agreement.update(perform_agreement)

    # Get the database agreement by averaging the agreement over all samples
    db_avg = np.average(list(db_agreement.values()))
    print("The IEMOCAP database has an emotion agreement percent of: "
          "{0:.2f}%.".format(db_avg * 100))


def _calculate_iem_perform_agree(perform_label_map):
    """
    Calculates the label/emotion agreement within an IEMOCAP performance.

    Sample output:
        {
            "Ses01F_impro01_F000": 0.375,
            "Ses01F_impro01_F001": 1.00, ...
        }

    :param perform_label_map: Dict mapping sample filenames to their emotions
    :return: Dict mapping sample filenames to their agreement percent
    """
    perform_agreement = {}

    for sample_name, sample_emotions in perform_label_map.items():
        # Discard which emotion was most voted for because we only care about
        # how many of the votes were convergent/divergent on an emotion
        _, emo_count = np.unique(sample_emotions, return_counts=True)
        perform_agreement[sample_name] = _calculate_iem_sample_agree(emo_count)

    return perform_agreement


def _calculate_iem_sample_agree(sample_emotions):
    """
    Calculates the label/emotion agreement within an IEMOCAP sample.

    Sample input:
        [2, 2, 3, 1]

    Sample output:
        0.375

    :param sample_emotions: List of votes for each emotion
    :return: Float representing the percent agreement
    """
    most_votes = np.max(sample_emotions)
    total_votes = np.sum(sample_emotions)

    sample_agreement = most_votes / total_votes
    return sample_agreement


def main():
    calculate_iemocap_agreement()


if __name__ == "__main__":
    main()
