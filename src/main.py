from src import preprocesser as pp
from src.database_loader import ravdess, cremad
import numpy as np


def main():
    """
    The main entry point to this program. This gets the data, preprocesses it
    for the machine learning model, submits it to the ML model, trains the ML
    model, and saves the results.
    """
    # Get the data
    ravdess_samples, ravdess_labels = ravdess.load_data()
    print(ravdess_samples.shape, ravdess_labels.shape)
    cremad_samples, cremad_labels = cremad.load_data()
    print(cremad_samples.shape, cremad_labels.shape)

    # Cache it into spectrograms

    # Create the spectrogram generator to feed the neural network

    # # Preprocess the data
    # processed_samples = pp.load_preprocess_samples(samples)
    # print(processed_samples.shape)
    # # processed_labels = pp.load_preprocess_labels(labels)
    # # print(processed_labels.shape)

    # Shuffle and create the train, validation, and testing sets

    # Create the ML model

    # Training the model

    # Save the model and training history

    # Test the model

    # Display results


if __name__ == "__main__":
    main()
