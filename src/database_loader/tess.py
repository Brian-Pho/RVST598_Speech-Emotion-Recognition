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

import db_constants as dbc
from src import constants as c

TES_BASE_URL = "https://tspace.library.utoronto.ca"
NUM_STD_CUTOFF = 3
TES_MIN_LEN, TES_MAX_LEN = None, None
TES_EMO_INDEX = 2
TES_SR = 24414
TES_EMOTION_MAP = {
    "neutral": c.NEU,
    "angry": c.ANG,
    "disgust": c.DIS,
    "fear": c.FEA,
    "happy": c.HAP,
    "sad": c.SAD,
    "ps": c.SUR,  # Map pleasant surprise to surprise
}
MEL_SPEC_FILENAME = "T_{id}_{emo_label}.npy"


def generate_stats():
    pass


def load_data():
    pass


def read_data():
    pass


def read_to_melspecgram():
    pass


def _interpret_label(filename):
    pass


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
    download process by automatically grabbing all of the wavs and saving
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
    # tess_samples, tess_labels = load_data()
    # print(tess_samples.shape)
    # print(tess_labels.shape)
    # generate_stats()
    # download_data()
    read_to_melspecgram()


if __name__ == "__main__":
    main()
