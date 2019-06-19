from src.database_loader import ravdess
from src import preprocesser as pp
import tensorflow as tf


def main():
    # Get the data
    ravdess_samples, ravdess_labels = ravdess.load_data()

    # Combine the datasets
    samples = ravdess_samples
    labels = ravdess_labels

    # Preprocess the data
    processed_samples = pp.preprocess_samples(samples)
    processed_labels = pp.preprocess_labels(labels)
    # print(labels.shape)
    # print(labels[::13])
    print(processed_labels.shape)
    # print(processed_labels[::20])

    # Shuffle and create the train, validation, and testing sets

    # Create the ML model

    # Training the model

    # Save the model and training history

    # Test the model

    # Display results


if __name__ == "__main__":
    main()
