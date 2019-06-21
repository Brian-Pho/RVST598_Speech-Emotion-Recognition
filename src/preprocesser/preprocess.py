import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from scipy import signal

from src import constants as c
from src.database_loader import db_constants as dbc


def load_preprocess_samples(samples):
    """
    Preprocesses the samples for use with a machine learning model. Loads the
    preprocessed samples from the cached files but if they don't exist, create
    them.

    :param samples: Tensor of shape (num_samples,)
    :return: Tensor of shape (num_samples,)
    """
    # Check if the preprocessed samples were cached
    preprocessed_samples = None

    try:
        preprocessed_samples = np.load(
            dbc.PRE_SAMPLES_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the preprocessed samples cache.")

    except IOError as e:
        print(str(e))
        preprocessed_samples = _preprocess_samples(samples)
        np.save(
            dbc.PRE_SAMPLES_CACHE_PATH, preprocessed_samples, allow_pickle=True)
        print("Successfully cached the preprocessed samples.")

    finally:
        return preprocessed_samples


def _preprocess_samples(samples):
    """
    Private function to preprocess the samples.

    :param samples: Tensor of shape (num_samples, 2)
    :return: Tensor of shape (num_samples, 2)
    """
    # Unpack the samples into two arrays, one of data and one of sampling rates
    data = np.array([sample[c.DATA_INDEX] for sample in samples])
    sr = np.array([sample[c.SR_INDEX] for sample in samples])

    # Normalize amplitudes
    data = _norm_amplitude(data)

    # Normalize durations
    data = _norm_durations(data)

    # Resample data to the highest sampling rate
    # samples = _norm_sampling_rates(samples)

    # Confirm that each sample has the same data shape and sampling rate
    if not _confirm_norm(data, sr):
        print("The dataset isn't consistent and an error occurred during the "
              "normalization of the samples.")
        return None
    else:
        print("The dataset is consistent.")

    # Convert into spectrogram
    samples = _convert_to_freq(data, sr)

    return np.array(list(zip(data, sr)))


def load_preprocess_labels(labels):
    """
    Preprocesses the labels for use with a machine learning model. Loads the
    preprocessed labels from the cached files but if they don't exist, create
    them.

    :param labels: Tensor of shape (num_samples,)
    :return: Tensor of shape (num_samples, num_emotions, )
    """
    # Check if the preprocessed labels were cached
    preprocessed_labels = None

    try:
        preprocessed_labels = np.load(
            dbc.PRE_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the preprocessed labels cache.")

    except IOError as e:
        print(str(e))
        preprocessed_labels = _preprocess_labels(labels)
        np.save(
            dbc.PRE_LABELS_CACHE_PATH, preprocessed_labels, allow_pickle=True)
        print("Successfully cached the preprocessed labels.")

    finally:
        return preprocessed_labels


def _preprocess_labels(labels):
    """
    Private function to preprocess the labels.

    :param labels: Tensor of shape (num_samples,)
    :return: Tensor of shape (num_samples, num_emotions, )
    """
    # Confirm that all emotions are present
    unique_emotions, counts = np.unique(labels, return_counts=True)
    if len(unique_emotions) != c.NUM_EMOTIONS:
        print("The labels don't represent all emotion classes.")

    # Print the number of emotions per class to check for class imbalance
    unique_emotions = [c.INVERT_EMOTION_MAP[label] for label in unique_emotions]
    print(dict(zip(unique_emotions, counts)))

    # One-hot encode the labels for use with categorical crossentropy
    return to_categorical(labels, num_classes=c.NUM_EMOTIONS)


def _norm_amplitude(data):
    """
    Normalizes the amplitude of each audio sample. In audio terms, this
    makes no sound too loud or too quiet within each sample. Performs
    sample-level amplitude normalization.

    :param data: Time-series audio data.
    :return: Tensor
    """
    norm_amplitude_samples = []
    for sample in data:
        sample -= sample.mean(axis=0)  # Set the mean to 0
        sample /= sample.std(axis=0)  # Set the variance to 1
        norm_amplitude_samples.append(sample)

    return np.array(norm_amplitude_samples)


def _norm_durations(data):
    """
    Normalizes the duration/length of each audio sample. We pad all samples that
    are shorter than the longest sample in the dataset.

    :param data: Time-series audio data.
    :return: Tensor
    """
    AXIS_ZERO = 0

    durations = [sample.shape[AXIS_ZERO] for sample in data]
    longest_duration = np.amax(durations)

    norm_duration_samples = []
    for sample in data:
        duration_diff = longest_duration - sample.shape[AXIS_ZERO]

        if duration_diff != 0:
            # For the time series, pad the ending with zeros.
            sample = np.pad(sample, pad_width=(0, duration_diff),
                            mode='constant', constant_values=0)
        norm_duration_samples.append(sample)

    return np.array(norm_duration_samples)


def _norm_sampling_rates(samples):
    """
    Normalizes the sampling rates of each audio sample. We up/down sample.

    :param samples: Samples from all databases.
    :return: Tensor
    """
    # TODO
    return samples


def _confirm_norm(data, sr):
    """
    Confirms that all samples are normalized with the same duration and sampling
    rate.

    :param data: Time-series audio data.
    :param sr: Sampling rates of the audio data.
    :return: True if all samples are consistent in data shape and sampling rate.
    """
    # Data shape mismatch
    data_shapes = [sample.shape[0] for sample in data]
    unique = np.unique(data_shapes)
    if len(unique) > 1:
        print("The data shapes for the samples aren't consistent.")
        return False

    # Sampling rate mismatch
    unique = np.unique(sr)
    if len(unique) > 1:
        print("The sampling rates for the samples aren't consistent.")
        return False

    # If we've gone through all of the samples and haven't returned, then all
    # samples are consistent.
    return True


def _convert_to_freq(data, sr):
    """
    Converts each audio sample from the time domain to the frequency domain.
    Uses the Short-Time Fourier Transform.

    :param data:
    :param sr:
    :return:
    """
    freq_dom_repr = []

    for data, sr in zip(data, sr):
        f, t, Zxx = signal.stft(data, sr, nperseg=c.WIN_SIZE,
                                noverlap=c.NUM_OVERLAP)

        # print(np.unwrap(np.angle(Zxx)))
        amplitude = np.log10(np.abs(Zxx))
        amplitude[amplitude == np.NINF] = np.amin(amplitude[amplitude != np.NINF])
        plt.pcolormesh(t, f, amplitude)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        plt.pcolormesh(t, f, np.unwrap(np.angle(Zxx)))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        freq_dom_repr.append((f, t, Zxx))

    return np.array(freq_dom_repr)
