# Database-level amplitude normalization
# z-score


def scale_sample(sample):
    """
    Scales a sample to be between 0 and 1.

    :param sample:
    :return:
    """
    pass


def norm_sample_lvl(db):
    """
    Normalize the database at the sample level. This means .

    :return: Tensor
    """
    pass


def norm_db_lvl(db):
    """
    Normalize the database at the database level. This means we calculate the
    mean and variance from all samples and apply it to each sample.

    :return: Tensor
    """
    db -= db.mean(axis=0)  # Set the mean to 0
    db /= db.std(axis=0)  # Set the variance to 1
    return db


# Convert time domain to frequency domain
# fourier transform

# Standard time length
# Padding/cropping

# Resample data to the highest sampling rate


def main():
    """
    Local testing of preprocessing.
    """



if __name__ == "__main__":
    main()
