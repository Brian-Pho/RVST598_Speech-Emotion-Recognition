"""
This file deals with downloading, processing, and loading the TESS database.
There are two options to use this database:
    - Load the database into RAM as a numpy array
    - Process the database into .npy files of log-mel spectrogram

Use the first option if you want quick access to the database or if you need to
apply operations to the entire database such as statistics.

Use the second option if you want to use your disk space instead of RAM and if
you plan on using a data generator to feed a neural network the samples.
"""

import os
from html.parser import HTMLParser
from urllib.request import urlopen, urlretrieve

import matplotlib.pyplot as plt
import numpy as np

import db_constants as dbc
from db_common import generate_db_stats, get_label
from src import em_constants as emc
from src.audio_processor.wav import load_wav, process_wav

TES_BASE_URL = "https://tspace.library.utoronto.ca"
TES_MIN_LEN, TES_MAX_LEN = None, None
TES_EMO_INDEX = 2
TES_SR = 24414
TES_EMOTION_MAP = {
    "neutral": emc.NEU,
    "angry": emc.ANG,
    "disgust": emc.DIS,
    "fear": emc.FEA,
    "happy": emc.HAP,
    "sad": emc.SAD,
    "ps": emc.SUR,  # Map pleasant surprise to surprise
}
MEL_SPEC_FILENAME = "T_{id}_{emo_label}.npy"


def load_data():
    """
    Loads the TESS database from the cached files. If they don't exist,
    then read the database and create the cache.

    :return: Tuple of (samples, labels) or None if unsuccessful.
    """
    tess_samples = None
    tess_labels = None

    try:
        # Attempt to read the cache
        tess_samples = np.load(dbc.TES_SAMPLES_CACHE_PATH, allow_pickle=True)
        tess_labels = np.load(dbc.TES_LABELS_CACHE_PATH, allow_pickle=True)
        print("Successfully loaded the TESS cache.")

    except IOError as e:
        # Since the cache doesn't exist, create it.
        print(str(e))
        tess_samples, tess_labels = read_data()
        np.save(dbc.TES_SAMPLES_CACHE_PATH, tess_samples, allow_pickle=True)
        np.save(dbc.TES_LABELS_CACHE_PATH, tess_labels, allow_pickle=True)
        print("Successfully cached the TESS database.")

    finally:
        # Pack the data
        ravdess_data = (tess_samples, tess_labels)
        return ravdess_data


def read_data():
    """
    Reads the TESS database into tensors.

    Sample output:
        (array([ 3.0517578e-05,  3.0517578e-05,  3.0517578e-05, ...,
        0.0000000e+00, -3.0517578e-05,  0.0000000e+00], dtype=float32), 0)

    :return: Tuple of (samples, labels) where the samples are a tensor of
             varying shape due to varying audio lengths, and the labels are an
             array of integers.
    """
    samples = []
    labels = []
    wav_folder = os.path.join(dbc.TES_DB_PATH, "data")

    for sample_filename in os.listdir(wav_folder):
        print("Processing file:", sample_filename)

        # Read the label and if it's empty, then drop the sample
        label = get_label(sample_filename, "_", TES_EMO_INDEX, TES_EMOTION_MAP)
        if not label:
            print("Not using sample:", sample_filename)
            continue

        # Read the sample
        sample_path = os.path.join(wav_folder, sample_filename)
        samples.append(load_wav(sample_path))

        labels.append(label)

    return np.array(samples), np.array(labels)


def read_to_melspecgram():
    """
    Reads the raw waveforms and converts them into log-mel spectrograms which
    are stored. This is an alternative to read_data() and load_data() to prevent
    using too much ram. Trades RAM for disk space.
    """
    id_counter = 0

    wav_folder = os.path.join(dbc.TES_DB_PATH, "data")

    for sample_filename in os.listdir(wav_folder):
        sample_path = os.path.join(wav_folder, sample_filename)

        # Read the sample
        wav = load_wav(sample_path)

        # No outlier check because all samples are within 3 stds

        # Process the sample into a log-mel spectrogram
        melspecgram = process_wav(wav)

        plt.pcolormesh(melspecgram, cmap="magma")
        plt.show()

        # Read the label
        label = get_label(sample_filename, "_", TES_EMO_INDEX, TES_EMOTION_MAP)

        # Save the log-mel spectrogram to use later
        mel_spec_path = os.path.join(
            dbc.PROCESS_DB_PATH, MEL_SPEC_FILENAME.format(
                id=id_counter, emo_label=label))
        np.save(mel_spec_path, melspecgram, allow_pickle=True)
        id_counter += 1


class TessHtmlParser(HTMLParser):
    """
    This class parses the TESS webpage for the wav file urls. We parse the
    web page to download the wav files because TESS doesn't provide any easy way
    to download all of the wav files.
    """
    def error(self, message):
        """
        Used to suppress PyCharm error.
        :param message: the error message during parsing
        """
        pass

    def __init__(self):
        """
        Class initializer.
        """
        HTMLParser.__init__(self)
        # E.g. ['/bitstream/1807/24488/1/OAF_youth_neutral.wav', ...]
        self.relative_wav_urls = []

    def handle_starttag(self, tag, attrs):
        """
        Handles when the parser encounters the start of an HTML tag.

        :param tag: the name of the tag
        :param attrs: the attributes of the tag
        """
        if tag != "a":
            return

        for name, value in attrs:
            if name == "href" and value.endswith(".wav"):
                self.relative_wav_urls.append(value)


def download_data():
    """
    Downloads the TESS database from the internet. This function automates the
    download process by automatically grabbing all of the wav files and saving
    them to the disk.
    """
    # 24488 to 24502 are numbers that are appended to the end of the url
    for handle_num in range(24488, 24502):
        handle_url = "{base}/handle/1807/{num}".format(
            base=TES_BASE_URL, num=handle_num)
        print("Processing url:", handle_url)

        # Open the url and read the HTML page
        web_stream = urlopen(handle_url)
        tess_page = str(web_stream.read())

        # Parse the HTML page for urls to wav files
        parser = TessHtmlParser()
        parser.feed(tess_page)

        # Expect 200 files per handle
        if len(parser.relative_wav_urls) != 200:
            print("An error occurred when parsing the TESS webpage for wav "
                  "files.")

        # Access the wav urls and save the wav files to the disk
        for wav_url in parser.relative_wav_urls:
            absolute_wav_url = "{base}{relative_url}".format(
                base=TES_BASE_URL, relative_url=wav_url)

            wav_filename = wav_url.split("/")[-1]
            wav_save_path = os.path.join(dbc.TES_DB_PATH, "data", wav_filename)
            urlretrieve(url=absolute_wav_url, filename=wav_save_path)

        print("Saved {} files.".format(len(parser.relative_wav_urls)))


def main():
    """
    Local testing and cache creation.
    """
    tess_samples, tess_labels = load_data()
    # print(tess_samples.shape)
    # print(tess_labels.shape)
    generate_db_stats(tess_samples, tess_labels)
    # download_data()
    # read_to_melspecgram()


if __name__ == "__main__":
    main()
