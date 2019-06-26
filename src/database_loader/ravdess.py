import os

import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from common import load_wav, calculate_bounds
from src import constants as c

NUM_ACTORS = 24
NUM_STD_CUTOFF = 3  # The number of standard deviations to include in the data
RAV_EMO_INDEX = 2  # The index into the filename for the emotion label
RAV_SR = 48000  # The sampling rate for all Ravdess audio samples
RAV_EMOTION_MAP = {
    "01": c.NEU,
    "02": c.NEU,  # Map calm to neutral
    "03": c.HAP,
    "04": c.SAD,
    "05": c.ANG,
    "06": c.FEA,
    "07": c.DIS,
    "08": c.SUR,
}


def generate_stats():
    """
    Generates statistics about the RAVDESS database.
    """
    # Convert each label back into an emotion.
    ravdess_samples, ravdess_labels = load_data()
    ravdess_labels = [c.INVERT_EMOTION_MAP[label] for label in ravdess_labels]

    # # Calculate the emotion class percentages. The neutral class has the most
    # # samples due to combining it with the calm class.
    # unique, counts = np.unique(ravdess_labels, return_counts=True)
    # print(dict(zip(unique, counts)))
    # plt.pie(x=counts, labels=unique)
    # plt.show()

    # Calculate the distribution of tensor shapes for the samples
    audio_lengths = [len(ts) for ts in ravdess_samples]

    lower, upper = calculate_bounds(audio_lengths, NUM_STD_CUTOFF)
    num_outliers = [length for length in audio_lengths
                    if length < lower or length > upper]
    print("Num outliers:", len(num_outliers))
    audio_cropped_lengths = [length for length in audio_lengths
                             if lower <= length <= upper]
    print("Num included:", len(audio_cropped_lengths))
    unique, counts = np.unique(audio_cropped_lengths, return_counts=True)

    data_min = unique[0]
    data_max = unique[-1]
    print(ravdess_samples.shape, data_min, data_max)

    plt.bar(unique, counts, width=800)
    plt.xlabel("Number of Data Points")
    plt.ylabel("Number of Samples")
    plt.title("The Distribution of Samples with Number of Data Points")
    plt.show()


def load_data():
    """
    Loads the RAVDESS database from the cached files. If they don't exist,
    then read the database and create the cache.

    :return: Tuple of (samples, labels) or None if unsuccessful.
    """
    ravdess_samples = None
    ravdess_labels = None

    try:
        # Attempt to read the cache
        ravdess_samples = np.load(dbc.RAV_SAMPLES_CACHE_PATH, allow_pickle=True)
        ravdess_labels = np.load(dbc.RAV_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the RAVDESS cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        ravdess_samples, ravdess_labels = read_data()
        np.save(dbc.RAV_SAMPLES_CACHE_PATH, ravdess_samples, allow_pickle=True)
        np.save(dbc.RAV_LABELS_CACHE_PATH, ravdess_labels, allow_pickle=True)
        print("Successfully cached the RAVDESS database.")

    finally:
        # Pack the data
        ravdess_data = (ravdess_samples, ravdess_labels)
        return ravdess_data


def read_data():
    """
    Reads the RAVDESS database into tensors.

    Sample output:
        (array([ 3.0517578e-05,  3.0517578e-05,  3.0517578e-05, ...,
        0.0000000e+00, -3.0517578e-05,  0.0000000e+00], dtype=float32), 0)

    :return: Tuple of (samples, labels) where the samples are a tensor of
             varying shape due to varying audio lengths, and the labels are an
             array of integers. The shape is (1440,) from 1440 audio files.
    """
    samples = []
    labels = []

    for actor in range(1, NUM_ACTORS + 1):
        actor_foldername = "Actor_{:02d}".format(actor)
        actor_path = os.path.join(dbc.RAV_DB_PATH, actor_foldername)
        print("Processing actor:", actor_foldername)

        for sample_filename in os.listdir(actor_path):
            sample_path = os.path.join(actor_path, sample_filename)

            # Read the sample
            audio_ts = load_wav(sample_path)

            # Remove the first and last second
            audio_ts = audio_ts[RAV_SR:-RAV_SR]

            samples.append(audio_ts)

            # Read the label
            labels.append(_interpret_label(sample_filename))

    return np.array(samples), np.array(labels)


def _interpret_label(filename):
    """
    Given a filename, it returns an integer representing the emotion label of
    the file/sample.

    :return: Integer
    """
    # Parse emotion ID from filename. It's the third number from the left
    # according to https://zenodo.org/record/1188976.
    emotion_id = filename.split("-")[RAV_EMO_INDEX]
    emotion = RAV_EMOTION_MAP[emotion_id]

    # Return a new emotion ID that's standardized across databases.
    return c.EMOTION_MAP[emotion]


def main():
    """
    Local testing and cache creation.
    """
    # ravdess_samples, ravdess_labels = load_data()
    # print((ravdess_samples[10], ravdess_labels[10]))
    # print(ravdess_samples.shape)
    # for sample in ravdess_samples[0:10]:
    #     print(sample.shape)
    # print(ravdess_labels.shape)
    # generate_stats()


if __name__ == "__main__":
    main()
