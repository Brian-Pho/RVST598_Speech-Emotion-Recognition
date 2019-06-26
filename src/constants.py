# The list of seven emotions considered. Based on Ekman, 1999.
NEU = "NEUTRAL"
ANG = "ANGER"
DIS = "DISGUST"
FEA = "FEAR"
HAP = "HAPPY"
SAD = "SAD"
SUR = "SURPRISE"

# The standard index for each emotion. This is how we keep the encoding of
# emotion consistent across databases. We do this because it'll help the
# neural network learn which tensor maps to which emotion. This will be useful
# in Keras's "to_categorical()" function.
# E.g. Anger maps to 1 and surprise maps to 6.
EMOTION_MAP = {
    NEU: 0,
    ANG: 1,
    DIS: 2,
    FEA: 3,
    HAP: 4,
    SAD: 5,
    SUR: 6,
}
# The inverted map is useful for interpreting the output of the neural network.
INVERT_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

NUM_EMOTIONS = len(EMOTION_MAP.keys())

# Training, validation, and testing allocations
TRAIN_ALLOC = 0.8
VALID_ALLOC = 0.1
TEST_ALLOC = 0.1

# Standard sampling rate
SR = 48000

# Short Time Fourier Transform (STFT) parameters
WIN_SIZE = 3072
STEP_SIZE = int(WIN_SIZE * 0.25)
NUM_OVERLAP = WIN_SIZE - STEP_SIZE
