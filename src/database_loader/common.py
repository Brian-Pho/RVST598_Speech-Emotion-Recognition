"""
This file holds common functions across all database processing. Functions
include loading and processing wav files.
"""

import librosa
import numpy as np

from src import constants as c
from src import specgram_helper as sgh

TIME_SERIES_DATA_TYPE = "float32"


def load_wav(wav_path):
    """
    Loads a wav file as a numpy array. Wraps the Librosa load function to keep
    the parameters consistent. Drops the sampling rate data because all wav's
    will be resampled to the sampling rate defined in constants.py.

    :param wav_path: Path to wav file
    :return: Tensor
    """
    audio_ts, sr = librosa.load(
        wav_path, sr=c.SR, dtype=np.dtype(TIME_SERIES_DATA_TYPE),
        res_type="kaiser_best")

    if sr != c.SR:
        print("Sampling rate mismatch.")
        return None

    return audio_ts


def process_wav(wav):
    """
    Processes a wav sample into a constant length, scaled, log-mel spectrogram.

    :param wav: The audio time series
    :return: np.array
    """
    # Pad to the constant length
    padded_wav = pad_wav(wav)

    # Convert to log-mel spectrogram
    melspecgram = sgh.wave_to_melspecgram(padded_wav)

    # Scale the spectrogram to be between -1 and 1
    scaled_melspecgram = sgh.scale_melspecgram(melspecgram)

    return scaled_melspecgram


def remove_first_last_sec(audio_ts, sr):
    """
    Removes the first and last second of an audio waveform.

    :param audio_ts: The audio time series data points
    :param sr: The sampling rate of the audio
    :return: Tensor
    """
    return audio_ts[sr:-sr]


def pad_wav(audio_ts, desired_length=c.MAX_DATA_POINTS):
    """
    Pads an audio waveform to the desired length by adding 0s to the right end.

    :param audio_ts: The audio time series data points
    :param desired_length: The desired length to pad to
    :return: Tensor
    """
    length_diff = desired_length - audio_ts.shape[0]
    audio_ts_padded = np.pad(audio_ts, pad_width=(0, length_diff),
                             mode='constant', constant_values=0)

    if len(audio_ts_padded) != desired_length:
        print("An error occurred during padding the waveform.")

    return audio_ts_padded


def calculate_bounds(data, num_std):
    """
    Calculates the lower and upper bound given a distribution and standard
    deviation.

    :param data: The dataset/distribution
    :param num_std: The number of standard deviations to set the bounds
    :return: Tuple, of the lower and upper bound
    """
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * num_std
    lower, upper = data_mean - cut_off, data_mean + cut_off
    return lower, upper


def is_outlier(audio_ts, lower, upper):
    """
    Checks if an audio sample is an outlier. Bounds are inclusive.

    :param audio_ts: The audio time series data points
    :param lower: The lower bound
    :param upper: The upper bound
    :return: Boolean
    """
    return False if lower <= len(audio_ts) <= upper else True
