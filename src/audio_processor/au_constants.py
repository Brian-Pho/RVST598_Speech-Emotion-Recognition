"""
This file holds audio constants such as standard sampling rates and wav data
type.
"""

# Standard sampling rate
SR = 48000

# Standard number of data points. Determined by the maximum data points in all
# database (the longest file) after cutting off samples outside of three std.
MAX_DATA_POINTS = 193794

# Standard data type for the time series / wav data
WAV_DATA_TYPE = "float32"

# Short Time Fourier Transform (STFT) parameters
WIN_SIZE = 3072
STEP_SIZE = int(WIN_SIZE * 0.25)
OVERLAP_SIZE = WIN_SIZE - STEP_SIZE

# Mel scale parameters. Determined by the human hearing range and what makes the
# spectrogram look good.
MIN_HERTZ = 20.0
MAX_HERTZ = 12000.0
NUM_MEL_BINS = 200  # Controls the y-axis resolution
