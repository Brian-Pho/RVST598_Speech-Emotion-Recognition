"""
This file holds the entry point into this program. This loads the data and
trains the ML model.
"""

import numpy as np

from src.neural_network import (
    nn_constants as nnc, nn_model as nnm, data_gen as dg)


def main():
    """
    The main entry point of this program.
    """
    # Get the data filenames (log-mel spectrograms in the form of .npy files)
    sample_fns = dg.get_sample_filenames()
    np.random.shuffle(sample_fns)

    # Shuffle and create the train, validation, and test sets
    num_total_samples = len(sample_fns)
    num_test = int(num_total_samples * nnc.TEST_ALLOC)
    num_valid = int(num_total_samples * nnc.VALID_ALLOC)
    num_not_train = num_test + num_valid
    num_train = num_total_samples - num_not_train
    print("The total number of samples to be used for training:",
          num_total_samples)

    test_samples = sample_fns[:num_test]
    valid_samples = sample_fns[num_test:num_not_train]
    train_samples = sample_fns[num_not_train:]

    # Create the batch generators to feed the neural network
    test_gen = dg.batch_generator(test_samples, nnc.BATCH_SIZE)
    valid_gen = dg.batch_generator(valid_samples, nnc.BATCH_SIZE)
    train_gen = dg.batch_generator(train_samples, nnc.BATCH_SIZE)

    # Calculate how many batches fit into each set. Used by Keras to know when
    # an epoch is complete.
    test_steps = np.ceil(num_test / nnc.BATCH_SIZE)
    valid_steps = np.ceil(num_valid / nnc.BATCH_SIZE)
    train_steps = np.ceil(num_train / nnc.BATCH_SIZE)

    # Create and train the model
    model = nnm.build_model()
    # model.load_weights(nnc.MODEL_SAVE_PATH)
    history = model.fit_generator(
        generator=train_gen, steps_per_epoch=train_steps, epochs=nnc.NUM_EPOCHS,
        verbose=nnc.VERBOSE_LVL, validation_data=valid_gen,
        validation_steps=valid_steps
    )

    # for inputs, targets in train_gen:
    #     print(targets[0:1])
    #     nnm.visualize_heatmap_activation(model, inputs[0:1])

    # Save the model and training history
    model.save(nnc.MODEL_SAVE_PATH)

    # Display the training history
    nnm.visualize_train_history(history)

    # Test the model on the test set
    test_loss, test_acc = model.evaluate_generator(
        generator=test_gen, steps=test_steps)
    print("Test loss:", test_loss, "Test acc:", test_acc)


if __name__ == "__main__":
    main()
