import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from scipy import signal
from gansynth.lib import specgrams_helper as sh
import tensorflow as tf

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
    data = samples

    # Normalize durations
    data = _norm_durations(data)

    # Convert into spectrogram
    samples = _convert_to_freq(data)

    # Normalize amplitudes
    samples = _norm_amplitude(samples)

    # Confirm that each sample has the same data shape and is scaled
    if not _check_samples():
        pass

    return np.array(samples)


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
    are shorter than the longest sample in the dataset with zeros.

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


def _check_samples(data):
    """
    Confirms that all samples are normalized with the same duration.

    :param data: Time-series audio data.
    :return: True if all samples are consistent in data shape and sampling rate.
    """
    # Data shape mismatch
    data_shapes = [sample.shape[0] for sample in data]
    unique = np.unique(data_shapes)
    if len(unique) > 1:
        print("The data shapes for the samples aren't consistent.")
        return False

    # If we've gone through all of the samples and haven't returned, then all
    # samples are consistent.
    return True


def _convert_to_freq(data):
    """
    Converts each audio sample from the time domain to the frequency domain.
    Uses the Short-Time Fourier Transform and converts the spectrogram to be mel
    scaled.

    :param d:
    :param sr:
    :return:
    """
    freq_dom_repr = []

    for d in data:
        stfts = tf.signal.stft(
            d, frame_length=c.WIN_SIZE, frame_step=c.WIN_SIZE - c.NUM_OVERLAP,
            fft_length=c.WIN_SIZE)
        print(stfts.shape)
        spectrograms = tf.abs(stfts)
        plt.pcolormesh(spectrograms)
        plt.show()

        # spec_helper = sh.SpecgramsHelper()
        # spec_helper.waves_to_melspecgrams(waves=)

        # f, t, Zxx = signal.stft(d, sr, nperseg=c.WIN_SIZE,
        #                         noverlap=c.NUM_OVERLAP)
        # Zxx /= np.abs(Zxx)
        #
        # amplitude = np.log10(np.abs(Zxx))
        # amplitude[amplitude == np.NINF] = np.amin(amplitude[amplitude != np.NINF])
        # print(f.shape)
        # print(t.shape)
        # print(Zxx.shape)
        #
        # mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        #     num_mel_bins=100, num_spectrogram_bins=len(f), sample_rate=sr,
        #     lower_edge_hertz=np.min(f), upper_edge_hertz=np.max(f))
        # amplitude = np.matmul(amplitude, mel_matrix)
        # plt.pcolormesh(t, f, amplitude)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()

        # plt.pcolormesh(t, f, np.unwrap(np.angle(Zxx)))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()

        freq_dom_repr.append((f, t, Zxx))

    return np.array(freq_dom_repr)
