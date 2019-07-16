"""
This file holds common audio functions for processing time-domain audio in the
form of .wav files and frequency-domain audio in the form of log-mel
spectrograms.
"""

import librosa
import noisereduce as nr
import numpy as np

import au_constants as auc
import spectrogram as sg


def load_wav(wav_path):
    """
    Loads a wav file as a numpy array. Wraps the Librosa load function to keep
    the parameters consistent. Drops the file's sampling rate because all wav
    files will be resampled to the sampling rate defined in constants.py.

    :param wav_path: Path to wav file
    :return: np.array
    """
    wav, sr = librosa.load(
        wav_path, sr=auc.SR, dtype=np.dtype(auc.WAV_DATA_TYPE),
        res_type=auc.RES_ALGOR)

    if sr != auc.SR:
        print("Sampling rate mismatch.")
        return None

    return wav


def process_wav(wav, noisy=False):
    """
    Processes a wav sample into a constant length, scaled, log-mel spectrogram.

    :param wav: The audio time series
    :param noisy: Used if the data is known to be noisy
    :return: np.array
    """
    # Reshape to a constant length. Slice if too long, pad if too short
    if auc.MAX_DATA_POINTS < len(wav):
        wav = wav[:auc.MAX_DATA_POINTS]
    else:
        wav = pad_wav(wav)

    if noisy:
        noisy_part = wav[:auc.NOISY_DURATION]
        # noinspection PyTypeChecker
        wav = nr.reduce_noise(
            audio_clip=wav, noise_clip=noisy_part, verbose=False)

    # Convert to log-mel spectrogram
    melspecgram = sg.wave_to_melspecgram(wav)

    # Scale the spectrogram to be between -1 and 1
    scaled_melspecgram = sg.scale_melspecgram(melspecgram)

    return scaled_melspecgram


def remove_first_last_sec(wav, sr):
    """
    Removes the first and last second of an audio clip.

    :param wav: The audio time series data points
    :param sr: The sampling rate of the audio
    :return: np.array
    """
    return wav[sr:-sr]


def pad_wav(wav, desired_length=auc.MAX_DATA_POINTS):
    """
    Pads an audio waveform to the desired length by adding 0s to the right end.

    :param wav: The audio time series data points
    :param desired_length: The desired length to pad to
    :return: np.array
    """
    length_diff = desired_length - wav.shape[0]

    if length_diff < 0:
        print("The waveform is longer than the desired length.")
        return None

    wav_padded = np.pad(
        wav, pad_width=(0, length_diff), mode='constant', constant_values=0)

    if len(wav_padded) != desired_length:
        print("An error occurred during padding the waveform.")
        return None

    return wav_padded
