"""
This file holds the entry point into this program. This loads the data and
trains the ML model.
"""
import numpy as np

from src import constants as c
from src import data_gen as dg
from src import nn_model as nnm
from src.database_loader import db_constants as dbc


def main():
    """
    The main entry point to this program.
    """
    # Get the data filenames (log-mel spectrograms in the form of .npy files)
    sample_fns = dg.get_sample_filenames()
    np.random.shuffle(sample_fns)
    # print(log_mel_samples.shape)

    # Shuffle and create the train, validation, and test sets
    num_total_samples = len(sample_fns)
    num_test = int(num_total_samples * c.TEST_ALLOC)
    num_valid = int(num_total_samples * c.VALID_ALLOC)
    num_not_train = num_test + num_valid
    num_train = num_total_samples - num_not_train

    test_samples = sample_fns[:num_test]
    valid_samples = sample_fns[num_test:num_not_train]
    train_samples = sample_fns[num_not_train:]

    # Create the batch generators to feed the neural network
    test_gen = dg.batch_generator(test_samples, c.BATCH_SIZE)
    valid_gen = dg.batch_generator(valid_samples, c.BATCH_SIZE)
    train_gen = dg.batch_generator(train_samples, c.BATCH_SIZE)

    # Calculate how many batches fit into each set. Used by Keras to know when
    # an epoch is complete.
    test_steps = np.ceil(num_test / c.BATCH_SIZE)
    valid_steps = np.ceil(num_valid / c.BATCH_SIZE)
    train_steps = np.ceil(num_train / c.BATCH_SIZE)

    # Create and train the model
    model = nnm.build_model()
    model.load_weights(dbc.MODEL_SAVE_PATH)
    # history = model.fit_generator(
    #     generator=train_gen, steps_per_epoch=train_steps, epochs=1100, verbose=2,
    #     validation_data=valid_gen, validation_steps=valid_steps, use_multiprocessing=False
    # )

    for inputs, targets in train_gen:
        print(targets[0:1])
        nnm.visualize_heatmap_activation(model, inputs[0:1])

    # Save the model and training history
    model.save(dbc.MODEL_SAVE_PATH)

    # Display the training history
    # nnm.display_history(history)

    # Test the model on the test set
    test_loss, test_acc = model.evaluate_generator(
        generator=test_gen, steps=test_steps)
    print("Test loss:", test_loss, "Test acc:", test_acc)


if __name__ == "__main__":
    main()
