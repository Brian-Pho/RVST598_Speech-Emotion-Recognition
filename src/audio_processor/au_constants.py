"""
This file holds audio constants such as the standard sampling rate and the
standard wav data type.
"""

# Standard sampling rate. The number of data points per second.
SR = 48000

# Standard number of data points. Convert to seconds by dividing by 48,000 (the
# sampling rate).
MAX_DATA_POINTS = 216000  # Corresponds to 4.5 seconds

# Standard data type for wav data
WAV_DATA_TYPE = "float32"

# Resampling algorithm. Options include "kaiser_best", "kaiser_fast", "fft", or
# "polyphase".
RES_ALGOR = "kaiser_best"

# Short Time Fourier Transform (STFT) parameters
WIN_SIZE = 3072
STEP_SIZE = int(WIN_SIZE * 0.25)
OVERLAP_SIZE = WIN_SIZE - STEP_SIZE

# Mel scale parameters. Determined by the human hearing range and what makes the
# spectrogram look good.
MIN_HERTZ = 20.0
MAX_HERTZ = 12000.0
NUM_MEL_BINS = 200  # Controls the y-axis resolution

# The range of values that the amplitude of a log-mel spectrogram can take
AMP_RANGE = (-1, 1)

# The length of the noisy part of a sample
NOISY_DURATION = int(SR * 0.5)

# The final mel-spectrogram shape. The first value is the same as the number of
# Mel bins specified above.
MEL_SPECGRAM_SHAPE = (200, 278)  # 200 rows with 278 columns.
