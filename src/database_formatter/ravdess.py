import os

import librosa
import numpy as np
from db_constants import RAVDESS_DB_PATH
from src import constants as c

NUM_ACTORS = 24


def load_data():
    """
    Reads the RAVDESS database into tensors.

    Sample output:
        [
            ((data, fs), label),
            ((array([0.0000000e+00, ...,], dtype=float32), 22050), 6),
            ...
        ]

    :return: Tensor of shape (60, 2) where 60 is the number of samples and 2 is
             the sample and label.
    """
    samples = []
    labels = []
    # for actor in range(1, NUM_ACTORS + 1):
    for actor in range(1, 10):
        actor_foldername = "Actor_{:02d}".format(actor)
        actor_path = os.path.join(RAVDESS_DB_PATH, actor_foldername)

        for sample_filename in os.listdir(actor_path):
            sample_path = os.path.join(actor_path, sample_filename)

            # Read the sample
            print(sample_filename)
            samples.append(librosa.load(sample_path))

            # Read the label
            labels.append(_interpret_label(sample_filename))

    return np.array(samples), np.array(labels)


def _interpret_label(filename):
    """
    Given a filename, it returns a tensor representing the emotion label of the
    file/sample.
    :return: Integer
    """
    RAVDESS_EMOTION_MAP = {
        "01": c.NEU,
        "02": c.NEU,
        "03": c.HAP,
        "04": c.SAD,
        "05": c.ANG,
        "06": c.FEA,
        "07": c.DIS,
        "08": c.SUR,
    }

    # Parse emotion ID from filename. It's the third number from the left
    # according to https://zenodo.org/record/1188976.
    emotion_id = filename.split("-")[2]
    emotion = RAVDESS_EMOTION_MAP[emotion_id]

    # Return a new emotion ID that's standardized across databases.
    # print(emotion, c.EMOTION_MAP[emotion])
    return c.EMOTION_MAP[emotion]


def main():
    ravdess_samples, ravdess_labels = load_data()
    print(ravdess_samples.shape)
    print(ravdess_labels.shape)
    # print(ravdess_data)
    # print(ravdess_data[0][1])
    # for data in ravdess_data:
    #     print(data[1])
    # print(load_data())


if __name__ == "__main__":
    main()
