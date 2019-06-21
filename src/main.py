from src import preprocesser as pp
from src.database_loader import ravdess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np


def main():
    # Get the data
    ravdess_samples, ravdess_labels = ravdess.load_data()
    ravdess_samples = ravdess.remove_first_sec(ravdess_samples)

    # Combine the datasets
    samples = ravdess_samples
    labels = ravdess_labels

    # samples, labels = None, None

    # Preprocess the data
    processed_samples = pp.load_preprocess_samples(samples)
    print(processed_samples.shape)

    # Test displaying and playing a waveform
    file_num = 79
    data = processed_samples[file_num][0]
    data /= np.amax(data)
    sd.play(data, processed_samples[file_num][1], blocking=True)
    plt.figure()
    librosa.display.waveplot(data, sr=processed_samples[file_num][1])
    plt.show()

    # processed_labels = pp.load_preprocess_labels(labels)
    # print(processed_labels.shape)

    # Shuffle and create the train, validation, and testing sets

    # Create the ML model

    # Training the model

    # Save the model and training history

    # Test the model

    # Display results


if __name__ == "__main__":
    main()
