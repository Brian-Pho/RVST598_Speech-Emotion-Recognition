"""
This file holds neural network constants such as training settings and the input
shape.
"""

import os

from keras.metrics import categorical_accuracy

from src.database_processor import db_constants as dbc

# MODEL SAVE PATH
MODEL_SAVE_PATH = os.path.join(dbc.DATA_PATH, "model.h5")
MODEL_PLOT_PATH = os.path.join(dbc.DATA_PATH, "model.png")

# MODEL CONFIGURATION
INPUT_SHAPE = (200, 278, 1)
OPTIMIZER = "rmsprop"
LOSS = "binary_crossentropy"
METRICS = [categorical_accuracy]

# TRAINING CONFIGURATION
NUM_EPOCHS = 10
BATCH_SIZE = 32
VERBOSE_LVL = 1
MIN_NUM_SAMPLES = 500

TRAIN_ALLOC = 0.8
VALID_ALLOC = 0.1
TEST_ALLOC = 0.1
