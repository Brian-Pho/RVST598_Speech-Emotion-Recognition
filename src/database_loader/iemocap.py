"""
This file deals with processing and loading the IEMOCAP database. There are two
options to use this database:
    - Load the database into RAM as a numpy array
    - Process the database into .npy files of log-mel spectrogram

Use the first option if you want quick access to the database or if you need to
apply operations to the entire database such as statistic.

Use the second option if you want to use your disk space instead of RAM and if
you plan on using a data generator to feed a neural network the samples.
"""

import db_constants as dbc
from common import load_wav, calculate_bounds, process_wav, is_outlier
from src import constants as c


def generate_stats():
    pass


def load_data():
    pass


def read_data():
    pass


def read_to_melspecgram():
    pass


def _interpret_label(filename):
    pass


def main():
    pass


if __name__ == "__main__":
    main()
