"""
This file holds neural network constants such as training settings and the input
shape.
"""

import os

from keras import optimizers, losses, metrics

from src.audio_processor import au_constants as auc
from src.database_processor import db_constants as dbc

# MODEL SAVE PATH
MODEL_SAVE_PATH = os.path.join(dbc.MODEL_PATH, "model.h5")
MODEL_PLOT_PATH = os.path.join(dbc.MODEL_PATH, "model.png")

# MODEL CONFIGURATION
NUM_CHANNELS = 1
INPUT_SHAPE = (
    auc.MEL_SPECGRAM_SHAPE[0], auc.MEL_SPECGRAM_SHAPE[1], NUM_CHANNELS)
OPTIMIZER = optimizers.adam()
LOSS = losses.binary_crossentropy
METRICS = [metrics.categorical_accuracy]

# TRAINING CONFIGURATION
NUM_EPOCHS = 10
NUM_WORKERS = 4
BATCH_SIZE = 32
VERBOSE_LVL = 1
MIN_NUM_SAMPLES = 500

TRAIN_ALLOC = 0.8
VALID_ALLOC = 0.1
TEST_ALLOC = 0.1
