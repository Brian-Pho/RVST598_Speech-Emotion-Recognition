import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from playsound import playsound


def main():
    fs, data = io.wavfile.read('../data/OAF_youth_happy.wav')
    f, t, Zxx = signal.stft(data, fs, nperseg=512, noverlap=300)
    plt.pcolormesh(t, f, np.log10(np.abs(Zxx)))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    print(data.shape)
    playsound('../data/OAF_youth_happy.wav')


if __name__ == "__main__":
    main()
