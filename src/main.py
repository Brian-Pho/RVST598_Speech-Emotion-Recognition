"""
This file holds the entry point into this program. This loads the data and
trains the ML model.
"""

import time

import numpy as np

from src.neural_network import (
    nn_constants as nnc, nn_model as nnm, data_gen as dg)


def main():
    """
    The main entry point of this program.
    """
    # Used to keep the shuffling of samples consistent between runs
    np.random.seed(0)
    # Get the samples (log-mel spectrograms in the form of .npy files)
    sample_fns = dg.get_sample_filenames()
    np.random.shuffle(sample_fns)  # Shuffle the dataset

    # Calculate the number of samples per set (train, valid, test)
    num_total_samples = len(sample_fns)
    num_test = int(num_total_samples * nnc.TEST_ALLOC)
    num_valid = int(num_total_samples * nnc.VALID_ALLOC)
    num_not_train = num_test + num_valid
    print("The number of samples for training:",
          num_total_samples - num_not_train)
    print("The number of samples for validation:", num_valid)
    print("The number of samples for testing:", num_test)
    print("The total number of samples:", num_total_samples)

    # Split the dataset into the train, valid, and test sets
    test_samples = sample_fns[:num_test]
    valid_samples = sample_fns[num_test:num_not_train]
    train_samples = sample_fns[num_not_train:]

    # Create the batch generators to feed the neural network
    test_gen = dg.BatchGenerator(test_samples)
    valid_gen = dg.BatchGenerator(valid_samples)
    train_gen = dg.BatchGenerator(train_samples)

    # Create and train the model
    model = nnm.build_model()
    # model.load_weights(nnc.MODEL_SAVE_PATH)
    class_weights = dg.get_class_weight(train_samples)
    history = nnm.train_model(model, train_gen, valid_gen, class_weights)

    # Save the model and training history
    model.save(nnc.MODEL_SAVE_PATH)
    nnm.save_history(history)

    # Display the training history
    nnm.visualize_train_history(history)

    # Display the confusion matrix for the test set
    nnm.visualize_confusion_matrix(model, test_gen)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate_generator(generator=test_gen)
    print("Test loss:", test_loss, "Test acc:", test_acc)


if __name__ == "__main__":
    main()
