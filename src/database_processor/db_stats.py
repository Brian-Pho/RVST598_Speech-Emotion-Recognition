"""
This file holds functions to calculate raw database statistics such as how many
outliers are outside a certain number of standard deviations and voter
agreement.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import cremad
import db_constants as dbc
import iemocap
from db_common import inverse_k_hot_encode_label
from src import em_constants as emc


def calculate_bounds(data, num_std=dbc.NUM_STD_CUTOFF):
    """
    Calculates the lower and upper bound given a distribution and standard
    deviation.

    :param data: The dataset/distribution
    :param num_std: The number of standard deviations to set the bounds
    :return: Tuple of the lower and upper bound
    """
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * num_std
    lower, upper = data_mean - cut_off, data_mean + cut_off
    return lower, upper


def generate_db_stats(samples, labels):
    """
    Generates statistics from the given samples and labels.

    :param samples: Samples from the database
    :param labels: Labels from the database
    """
    # Convert from k-hot encoded labels to emotion ID labels
    emo_labels = []
    for label in labels:
        label = inverse_k_hot_encode_label(label)
        emo_labels = np.concatenate([emo_labels, label])

    # Convert from emotion ID labels into the actual emotion
    unique, counts = np.unique(emo_labels, return_counts=True)
    unique = [emc.INVERT_EMOTION_MAP[label] for label in unique]

    print(dict(zip(unique, counts)))
    plt.pie(x=counts, labels=unique)
    plt.show()

    # Calculate the distribution of np.array shapes for the samples
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
    plt.title("The Distribution of Samples with Number of Data Points")
    plt.xlabel("Number of Data Points")
    plt.ylabel("Number of Samples")
    plt.show()


def calculate_cremad_accuracy():
    """
    Calculates the accuracy of the CREMA-D databse. Accuracy is defined as the
    percent agreement between the intended emotion and the perceived emotion.

    Example:
        Suppose a sample was spoken with the intent of conveying the emotion
        "Anger" but listeners voted the sample as ["Anger", "Anger", "Disgust",
        "Fear"]. Then the accuracy for this sample is 50% because
        only 50% of the voters got the "right" answer.

    Expected output:
        "The CREMA-D database has an emotion accuracy percent of: 61.89%."
    """
    labels_path = os.path.join(dbc.CRE_DB_PATH, "tabulatedVotes.csv")
    label_map = cremad.get_label_map(labels_path, encode=False)
    sample_accs = 0  # List of sample accuracies

    for sample_name, sample_emotions in label_map.items():
        # Get the intended emotion from the sample's filename
        intended_emotion = sample_name.split("_")[2][0]

        # Get the perceived emotions and their respective votes
        perceived_emotions, emotion_votes = np.unique(
            sample_emotions, return_counts=True)

        # If there isn't a match, then assume this sample has an accuracy of
        # zero.
        if intended_emotion not in perceived_emotions:
            continue

        # If there is a match, calculate how strong the match is by getting the
        # votes for the accurate emotion divided by the total votes
        perceived_emo_index = np.where(intended_emotion == perceived_emotions)
        sample_acc = emotion_votes[perceived_emo_index] / len(sample_emotions)
        sample_accs += sample_acc

    cremad_acc = np.sum(sample_accs) / len(label_map)
    print("The CREMA-D database has an emotion accuracy percent of: "
          "{0:.2f}%.".format(cremad_acc * 100))


def calculate_iemocap_agreement():
    """
    Calculates the emotion agreement percentage of the IEMOCAP database.

    Expected output:
        "The IEMOCAP database has an emotion agreement percent of: 63.60%."
    """
    db_agreement = {}
    labels_path = os.path.join(dbc.IEM_DB_PATH, "labels")

    for num_sess in range(1, iemocap.NUM_SESS + 1):
        sess_foldername = "S{}".format(num_sess)
        print("Processing session:", sess_foldername)

        sess_labels_path = os.path.join(labels_path, sess_foldername)

        for perform in os.listdir(sess_labels_path):
            perform_labels_path = os.path.join(sess_labels_path, perform)

            # Get the agreement percents for each sample in this performance
            perform_label_map = iemocap.get_label_map(
                perform_labels_path, encode=False)
            perform_agreement = _calc_iem_perform_agree(perform_label_map)
            db_agreement.update(perform_agreement)

    # Get the database agreement by averaging the agreement over all samples
    db_avg = np.average(list(db_agreement.values()))
    print("The IEMOCAP database has an emotion agreement percent of: "
          "{0:.2f}%.".format(db_avg * 100))


def _calc_iem_perform_agree(perform_label_map):
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
        perform_agreement[sample_name] = _calc_iem_sample_agree(emo_count)

    return perform_agreement


def _calc_iem_sample_agree(sample_emotions):
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
    calculate_cremad_accuracy()
    calculate_iemocap_agreement()


if __name__ == "__main__":
    main()
