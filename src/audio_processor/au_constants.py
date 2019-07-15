"""
This file holds audio constants such as standard sampling rates and wav data
type.
"""

# Standard sampling rate. The number of data points per second.
SR = 48000

# Standard number of data points. Determined by the maximum data points in all
# database (the longest file). Convert to seconds by dividing by 48,000 (the
# sampling rate).
MAX_DATA_POINTS = 216000  # Corresponds to 4.5 seconds

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

# The range of values that a log-mel spectrogram can take
AMP_RANGE = (-1, 1)

# The length of the noisy part of a sample
NOISY_DURATION = int(SR * 0.5)
