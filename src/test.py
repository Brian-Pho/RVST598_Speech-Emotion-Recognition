import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from playsound import playsound
from scipy import signal

from src.database_loader import ravdess


def main():
    fs, data = io.wavfile.read('../data/OAF_youth_happy.wav')
    nperseg =
    f, t, Zxx = signal.stft(data, fs, nperseg=512, noverlap=300)
    plt.pcolormesh(t, f, np.log10(np.abs(Zxx)))
    # plt.pcolormesh(t, f, np.unwrap(np.angle(Zxx)))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    print(data.shape)
    playsound('../data/OAF_youth_happy.wav')

    start_time = time.time()
    ravdess_samples, ravdess_labels = ravdess.load_data()
    end_time = time.time()
    print("Database load time:", end_time - start_time)

# preprocess the data by
# - normalizing amplitude and file length
# - to_categorical the labels
# - convert to spec gram


if __name__ == "__main__":
    main()
