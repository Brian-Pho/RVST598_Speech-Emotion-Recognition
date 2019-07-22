"""
This file holds functions to calculate the final processed database statistics
such as how many times an emotion is present and how many files came from
each database.
"""

import os

import numpy as np

from src.database_processor import db_constants as dbc
from src.neural_network import data_gen as dg


def calc_emotion_class(samples):
    """
    Calculates the number of samples for each emotion class.

    :param samples: A list of sample filenames

    Sample output:

    """
    emotion_class_counts = np.zeros(7, dtype=int)

    for sample in samples:
        label = dg.read_label(sample)
        emotion_class_counts += label

    print(emotion_class_counts)


def calc_label_type(samples):
    """
    Calculates the number of labels that fall into a label type. Label types
    include no label, one label, or multi-label.

    :param samples: A list of sample filenames

    Sample output:

    """
    label_type_map = {}

    for sample in samples:
        label = dg.read_label(sample)
        num_labels = np.sum(label)

        if num_labels not in label_type_map.keys():
            label_type_map[num_labels] = 0

        label_type_map[num_labels] += 1

    print(label_type_map)


def count_samples_per_db(samples):
    """
    Counts the number of samples per database.

    :param samples: A list of sample filenames

    Sample output:

    """
    db_counts = {}

    for sample in samples:
        db = sample.split("-")[0].split("_")[0]

        if db not in db_counts.keys():
            db_counts[db] = 0

        db_counts[db] += 1

    print(db_counts)


def main():
    sample_fns = os.listdir(dbc.PROCESS_DB_PATH)
    calc_emotion_class(sample_fns)
    calc_label_type(sample_fns)
    count_samples_per_db(sample_fns)


if __name__ == "__main__":
    main()
