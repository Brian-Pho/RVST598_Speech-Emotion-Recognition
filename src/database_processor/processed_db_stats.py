"""
This file holds functions to calculate the final processed database statistics
such as how many times an emotion is present and how many files came from
each database.
"""

import os

import numpy as np

from src import em_constants as emc
from src.database_processor import db_constants as dbc
from src.neural_network import data_gen as dg


def calc_emotion_class(samples):
    """
    Calculates the number of samples for each emotion class.

    :param samples: A list of sample filenames

    Sample output:
        Emotion class: {
            'NEUTRAL': 5994, 'ANGER': 6209, 'DISGUST': 1960, 'FEAR': 1884,
            'HAPPY': 4745, 'SAD': 3040, 'SURPRISE': 883
        }
    """
    emotion_running_counts = np.zeros(7, dtype=int)

    for sample in samples:
        label = dg.read_label(sample)
        emotion_running_counts += label

    emotion_total_counts = {}
    for index, counts in enumerate(emotion_running_counts):
        emotion_total_counts[emc.INVERT_EMOTION_MAP[index]] = counts

    print("Emotion class:", emotion_total_counts)
    print("Total:", sum(emotion_total_counts.values()))


def calc_label_type(samples):
    """
    Calculates the number of labels that fall into a label type. Label types
    include no label, one label, or multi-label.

    :param samples: A list of sample filenames

    Sample output:
        Label type map: {
            1: 19884, 2: 727, 3: 1064, 4: 45, 5: 1
        }
    """
    label_type_map = {}

    for sample in samples:
        label = dg.read_label(sample)
        num_labels = np.sum(label)

        if num_labels not in label_type_map.keys():
            label_type_map[num_labels] = 0

        label_type_map[num_labels] += 1

    print("Label type map:", label_type_map)
    print("Total:", sum(label_type_map.values()))


def count_samples_per_db(samples):
    """
    Counts the number of samples per database.

    :param samples: A list of sample filenames

    Sample output:
        Samples per db: {
            'CRE': 7442, 'IEM': 10039, 'RAV': 1440, 'TES': 2800
        }
    """
    db_counts = {}

    for sample in samples:
        db = sample.split("-")[0].split("_")[0]

        if db not in db_counts.keys():
            db_counts[db] = 0

        db_counts[db] += 1

    print("Samples per db:", db_counts)
    print("Total:", sum(db_counts.values()))


def remove_label_type(samples, label_types):
    """
    Removes samples from the processed database that have a certain label type.
    Label types are the number of labels a given sample has such as "2".

    :param samples: A list of sample filenames
    :param label_types: A list of label types to filter/remove
    """
    for sample in samples:
        label = dg.read_label(sample)
        num_labels = np.sum(label)

        if num_labels in label_types:
            sample_path = os.path.join(dbc.PROCESS_DB_PATH, sample)
            os.remove(sample_path)
            print("Removed sample:", sample)


def main():
    sample_fns = os.listdir(dbc.PROCESS_DB_PATH)
    remove_label_type(sample_fns, [4, 5])
    calc_emotion_class(sample_fns)
    calc_label_type(sample_fns)
    count_samples_per_db(sample_fns)


if __name__ == "__main__":
    main()
