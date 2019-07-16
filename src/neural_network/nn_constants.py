"""
This file holds neural network constants such as training settings and the input
shape.
"""

import os

from src.database_processor import db_constants as dbc

# MODEL SAVE PATH
MODEL_SAVE_PATH = os.path.join(dbc.DATA_PATH, "model.h5")

# MODEL CONFIGURATION
INPUT_SHAPE = (200, 278, 1)
OPTIMIZER = "adam"
LOSS = "binary_crossentropy"
METRICS = ['binary_accuracy', 'categorical_accuracy']

# TRAINING CONFIGURATION
NUM_EPOCHS = 100
BATCH_SIZE = 32
VERBOSE_LVL = 2
MIN_NUM_SAMPLES = 0

TRAIN_ALLOC = 0.8
VALID_ALLOC = 0.1
TEST_ALLOC = 0.1
