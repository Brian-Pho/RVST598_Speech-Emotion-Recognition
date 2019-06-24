import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from src import constants as c

NUM_ACTORS = 24
SAMPLES_THRESHOLD = 206000  # The max number of data points for a file
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

    # Calculate the emotion class percentages. The neutral class has the most
    # samples due to combining it with the calm class.
    unique, counts = np.unique(ravdess_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    plt.pie(x=counts, labels=unique)
    plt.show()

    # # Calculate the distribution of tensor shapes for the samples
    # ravdess_samples = remove_first_last_sec(ravdess_samples)
    # time_series = [len(ts[0]) for ts in ravdess_samples]
    # unique, counts = np.unique(time_series, return_counts=True)
    #
    # counts_above_threshold = 0
    # threshold = 110000
    # for u, count in zip(unique, counts):
    #     if u > threshold:
    #         counts_above_threshold += count
    # print("Percent above {} samples".format(threshold),
    #       counts_above_threshold / np.sum(counts))
    # cumulative = 0
    # cumulative_count = []
    # for number in counts:
    #     cumulative += number
    #     cumulative_count.append(cumulative)
    # print(dict(zip(unique, counts)))
    # min = time_series.index(unique[0])
    # max = time_series.index(unique[-1])
    # print(len(ravdess_samples[min][0]))
    # print(len(ravdess_samples[max][0]))
    # print(ravdess_labels[min])
    # print(ravdess_labels[max])

    # cumulative_count /= np.amax(cumulative_count)
    # plt.plot(np.arange(len(cumulative_count)), cumulative_count)
    # plt.bar(unique, counts, width=1000)
    # plt.xlabel("Number of Data Points")
    # plt.ylabel("Number of Samples")
    # plt.title("The Distribution of Samples with Certain Data Points")
    # plt.show()

    # # Calculate the distribution of data points for the first second which is
    # # the same as the first 48000 data points
    # first_sec_time_series = [ts[0][0:48000] for ts in ravdess_samples]
    # unique, counts = np.unique(first_sec_time_series, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(unique[0])
    # print(unique[-1])
    # plt.bar(unique, counts, width=0.1)
    # plt.show()

    # # Test playing and displaying a waveform
    # sd.play(ravdess_samples[0][0], ravdess_samples[0][1], blocking=True)
    # plt.figure()
    # librosa.display.waveplot(ravdess_samples[0][0], sr=ravdess_samples[0][1])
    # plt.show()


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
            audio_time_series, sampling_rate = librosa.load(sample_path, sr=None)
            if sampling_rate != RAV_SR:
                print("RAVDESS sampling rate mismatch.")
                continue
            samples.append(audio_time_series)  # Discard sampling rate

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


def remove_first_last_sec(rav_db):
    """
    Removes the first and last second (the first and last 48,000 data points
    because the audio is sampled at 48 kHz) from the Ravdess database. Most, if
    not all, samples in the Ravdess database have empty/quiet first and last
    seconds.

    :param rav_db: Samples from the Ravdess database
    :return: Tensor
    """
    processed_rav_db = []

    for sample in rav_db:
        # Slice the first and last second out of the data
        sliced_data = sample[RAV_SR:-RAV_SR]
        # Copy the sampling rate to the new database
        processed_rav_db.append(sliced_data)

    return np.array(processed_rav_db)


def remove_outliers(rav_db):
    """
    Removes samples that have data points above a defined threshold. In audio
    terms, removes samples that are too long (the top 6%).

    :param rav_db: Samples from the Ravdess database
    :return: Tensor
    """
    processed_rav_db = []

    for sample in rav_db:
        # If the sample is above the threshold, then skip it and don't add it.
        if sample.shape[0] > SAMPLES_THRESHOLD:
            continue
        processed_rav_db.append(sample)

    return np.array(processed_rav_db)


def main():
    """
    Local testing and cache creation.
    """
    ravdess_samples, ravdess_labels = load_data()
    print((ravdess_samples[10], ravdess_labels[10]))
    # print(ravdess_samples.shape)
    # for sample in ravdess_samples[0:10]:
    #     print(sample.shape)
    # print(ravdess_labels.shape)
    # generate_stats()


if __name__ == "__main__":
    main()
