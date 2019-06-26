import librosa
import numpy as np

from src import constants as c

TIME_SERIES_DATA_TYPE = "float32"


def load_wav(wav_path):
    """
    Loads a wav file as a numpy array. Wraps the Librosa load function to keep
    the parameters constant.

    :param wav_path: Path to wav file
    :return: Tensor
    """
    # Ignore sampling rate
    audio_ts, sr = librosa.load(
        wav_path, sr=c.SR, dtype=np.dtype(TIME_SERIES_DATA_TYPE),
        res_type="kaiser_best")

    if sr != c.SR:
        print("Sampling rate mismatch.")
        return None

    return audio_ts


def calculate_bounds(data, num_std):
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * num_std
    lower, upper = data_mean - cut_off, data_mean + cut_off
    return lower, upper
