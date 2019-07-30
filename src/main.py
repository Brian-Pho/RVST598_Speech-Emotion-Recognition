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
    # Get the samples (log-mel spectrograms in the form of .npy files)
    sample_fns = dg.get_sample_filenames()
    np.random.shuffle(sample_fns)  # Shuffle the dataset

    # Calculate the number of samples per set (train, valid, test)
    num_total_samples = len(sample_fns)
    num_test = int(num_total_samples * nnc.TEST_ALLOC)
    num_valid = int(num_total_samples * nnc.VALID_ALLOC)
    num_not_train = num_test + num_valid
    print("The total number of samples used:", num_total_samples)

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
    class_weights = dg.get_class_weight(train_samples)
    # model.load_weights(nnc.MODEL_SAVE_PATH)
    start_train_time = time.time()
    history = model.fit_generator(
        generator=train_gen, epochs=nnc.NUM_EPOCHS, verbose=nnc.VERBOSE_LVL,
        validation_data=valid_gen, class_weight=class_weights,
        use_multiprocessing=True, workers=nnc.NUM_WORKERS
    )
    end_train_time = time.time()

    # for inputs, targets in train_gen:
    #     print(targets[0:1])
    #     nnm.visualize_heatmap_activation(model, inputs[0:1])

    # Display the confusion matrix
    batch, true_out = test_gen[0]
    pred_out = model.predict_on_batch(batch)
    nnm.visualize_confusion_matrix(true_out, pred_out)


if __name__ == "__main__":
    main()
