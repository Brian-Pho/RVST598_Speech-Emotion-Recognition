import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

from db_constants import (
    RAV_RAW_DB_PATH, RAV_SAMPLES_CACHE_PATH, RAV_LABELS_CACHE_PATH)
from src import constants as c

NUM_ACTORS = 24
RAVDESS_EMOTION_MAP = {
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
        
    # Calculate the class percentages. The neutral class has the most samples
    # due to combining it with the calm class.
    # unique, counts = np.unique(ravdess_labels, return_counts=True)
    # print(dict(zip(unique, counts)))
    # plt.pie(x=counts, labels=unique)
    # plt.show()

    # Calculate the distribution of tensor shapes for the samples
    # time_series = [len(ts[0]) for ts in ravdess_samples]
    # unique, counts = np.unique(time_series, return_counts=True)
    # print(dict(zip(unique, counts)))
    # min = time_series.index(unique[0])
    # max = time_series.index(unique[-1])
    # print(len(ravdess_samples[min][0]))
    # print(len(ravdess_samples[max][0]))
    # print(ravdess_labels[min])
    # print(ravdess_labels[max])
    # plt.bar(unique, counts, width=1000)
    # plt.xlabel("Number of Data Points")
    # plt.ylabel("Number of Samples")
    # plt.title("The Distribution of Samples with Certain Data Points")
    # plt.show()

    # Calculate the distribution of data points for the first second which is
    # the same as the first 48000 data points
    # first_sec_time_series = [ts[0][0:48000] for ts in ravdess_samples]
    # unique, counts = np.unique(first_sec_time_series, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(unique[0])
    # print(unique[-1])
    # plt.bar(unique, counts, width=0.1)
    # plt.show()

    # Test displaying a waveform
    sd.play(ravdess_samples[0][0], ravdess_samples[0][1], blocking=True)
    plt.figure()
    librosa.display.waveplot(ravdess_samples[0][0], sr=ravdess_samples[0][1])
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
        ravdess_samples = np.load(RAV_SAMPLES_CACHE_PATH, allow_pickle=True)
        ravdess_labels = np.load(RAV_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the RAVDESS cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        ravdess_samples, ravdess_labels = read_data()
        np.save(RAV_SAMPLES_CACHE_PATH, ravdess_samples, allow_pickle=True)
        np.save(RAV_LABELS_CACHE_PATH, ravdess_labels, allow_pickle=True)
        print("Successfully cached the RAVDESS database.")

    finally:
        # Pack the data
        ravdess_data = (ravdess_samples, ravdess_labels)
        return ravdess_data


def read_data():
    """
    Reads the RAVDESS database into tensors.

    Sample output:
        (array([[array([ 1.5591205e-07, -1.5845627e-07,  1.5362870e-07, ...,
        0.0000000e+00,  0.0000000e+00,  0.0000000e+00], dtype=float32),
        22050]], dtype=object), array([0, 0, 0, ..., 6]))

    :return: Tuple of (samples, labels) where the samples are a tensor of shape
             (files, data), and the labels are an array of integers. The data is
             formatted as (audio time series, sampling rate).
    """
    samples = []
    labels = []
    for actor in range(1, NUM_ACTORS + 1):
        actor_foldername = "Actor_{:02d}".format(actor)
        actor_path = os.path.join(RAV_RAW_DB_PATH, actor_foldername)
        print(actor_foldername)

        for sample_filename in os.listdir(actor_path):
            sample_path = os.path.join(actor_path, sample_filename)

            # Read the sample
            samples.append(librosa.load(sample_path, sr=None))

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
    emotion_id = filename.split("-")[2]
    emotion = RAVDESS_EMOTION_MAP[emotion_id]

    # Return a new emotion ID that's standardized across databases.
    return c.EMOTION_MAP[emotion]


def remove_first_sec(rav_samples):
    """
    Removes the first second (the first 48,000 data points because the audio is
    sampled at 48 kHz) from the Ravdess database.

    :param rav_samples: Samples from the Ravdess database
    :return: Tensor
    """
    processed_rav_db = []

    for sample in rav_samples:
        # Slice the first second out of the data
        sr = sample[c.SR_INDEX]
        sliced_data = sample[c.DATA_INDEX][sr:]
        # Copy the sampling rate to the new database
        processed_rav_db.append((sliced_data, sample[c.SR_INDEX]))

    return np.array(processed_rav_db)


def main():
    """
    Local testing and cache creation.
    """
    ravdess_samples, ravdess_labels = load_data()
    print(ravdess_samples.shape)
    print(ravdess_samples[0][0].shape)  # Amplitude data
    ravdess_samples = remove_first_sec(ravdess_samples)
    print(ravdess_samples[0][0].shape)
    print(ravdess_samples[0][1])  # Sampling rate data
    # print(ravdess_labels.shape)
    # generate_stats()


if __name__ == "__main__":
    main()
