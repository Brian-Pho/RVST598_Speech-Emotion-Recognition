import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from db_common import inverse_k_hot_encode_label
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
