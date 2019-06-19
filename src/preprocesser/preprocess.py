from keras.utils import to_categorical
from src import constants as c
import numpy as np


def preprocess_samples(samples):
    pass


def preprocess_labels(labels):
    """
    Preprocesses the labels for use with a machine learning model.

    :param labels: Tensor of shape (num_samples,)
    :return: Tensor of shape
    """
    # Confirm that all emotions are present
    unique_emotions, counts = np.unique(labels, return_counts=True)
    if len(unique_emotions) != c.NUM_EMOTIONS:
        print("The labels don't represent all emotion classes.")

    # Print the number of emotions per class to check for class imbalance
    unique_emotions = [c.INVERT_EMOTION_MAP[label] for label in unique_emotions]
    print(dict(zip(unique_emotions, counts)))

    # One-hot encode the labels for us with categorical crossentropy
    return to_categorical(labels, num_classes=c.NUM_EMOTIONS)

# Database-level amplitude normalization
# z-score


def _scale_sample(sample):
    """
    Scales a sample to be between 0 and 1.

    :param sample:
    :return:
    """
    pass


def _norm_sample_lvl(db):
    """
    Normalize the database at the sample level. This means .

    :return: Tensor
    """
    pass


def _norm_db_lvl(db):
    """
    Normalize the database at the database level. This means we calculate the
    mean and variance from all samples and apply it to each sample.

    :return: Tensor
    """
    db -= db.mean(axis=0)  # Set the mean to 0
    db /= db.std(axis=0)  # Set the variance to 1
    return db


# Convert time domain to frequency domain
# - fourier transform

# Standardize time length
# - Padding/cropping

# Resample data to the highest sampling rate
