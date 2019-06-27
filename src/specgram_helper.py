"""
This file holds functions to help build audio spectrograms, specifically
log-mel spectrograms.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale

from src import constants as c

EPS = 1.0e-6  # Epsilon, a small value to avoid the log(0) problem.


def wave_to_melspecgram(wave):
    """
    Converts an audio waveform into a log-mel spectrogram. Based off of
    https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms

    :param wave: The time series of an audio clip
    :return: NumPy array of complex numbers
    """
    # Convert the wave into the frequency domain
    stft = tf.signal.stft(wave, frame_length=c.WIN_SIZE,
                          frame_step=c.STEP_SIZE)
    spectrogram = tf.abs(stft)

    # Warp the linear scale spectrogram into the mel-scale
    num_specgram_bins = stft.shape[-1].value
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        c.NUM_MEL_BINS, num_specgram_bins, c.SR, c.MIN_HERTZ,
        c.MAX_HERTZ)
    mel_specgram = tf.tensordot(
        spectrogram, linear_to_mel_weight_matrix, 1)
    mel_specgram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Flip the rows and columns so that the mel bins are on the y-axis and
    # time on the x-axis.
    mel_specgram = tf.transpose(mel_specgram)

    # Compute a stabilized log to get log-magnitude mel-scale spectrogram
    log_mel_specgram = tf.math.log(mel_specgram + EPS)

    sess = tf.compat.v1.Session()
    with sess.as_default():
        # Convert the spectrogram from a tf.Tensor into a np.array
        log_mel_specgram = log_mel_specgram.eval()

    return log_mel_specgram


def normalize_melspecgram(melspecgram):
    """
    Normalizes the amplitude of a log-mel spectrogram to have zero mean and
    unit variance. Performs sample-level amplitude normalization.

    :param melspecgram: np.array
    :return: np.array
    """
    melspecgram -= np.mean(melspecgram)  # Set the mean to 0
    melspecgram /= np.std(melspecgram)  # Set the variance to 1
    return melspecgram


def scale_melspecgram(melspecgram, amp_range=(-1, 1)):
    """
    Scales a log-mel spectrogram to have an amplitude range of [-1, 1]. Don't
    use both normalization and scaling.

    :param melspecgram: np.array
    :param amp_range: Tuple specifying the desired range
    :return: np.array
    """
    return minmax_scale(melspecgram, amp_range)
