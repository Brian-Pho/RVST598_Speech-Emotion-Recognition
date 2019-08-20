# RVST598_Speech-Emotion-Recognition

This is my summer (May - Aug) 2019 research project on using machine learning to detect emotions in speech.

## Plan

My goal is to detect which emotions are present in a speech sample. Specifically, this problem is called multi-class, multi-label speech emotion recognition. I consider these seven
emotions:

- Neutral
- Anger
- Disgust
- Fear
- Happy
- Sad
- Surprise

A speech sample is a few seconds long and flows through the following data processing pipeline:

1. Raw waveform (time + amplitude)
2. Spectrogram (time + frequency + ampltitude)
3. Log-Mel Spectrogram (time + log-mel + ampltitude)
4. Convolutional Neural Network (CNN)
5. Output (one or more of the seven emotions)

### Timeline

- May: Read the `Deep Learning with Python` textbook by Francois Chollet to pickup deep learning. My personal notes can be found [here](https://brianpho.com/CR4-DL/textbooks/2019/06/02/deep-learning-with-python.html).
- June: Decide on this project and start collecting and processing the databases.
- July: Process the databases and start creating the neural network model.
- August: Complete the neural network training, write the paper, and create the presentation.

## Databases Used

- IEMOCAP: Interactive Emotional Dyadic Motion Capture
- CREMA-D: Crowd-sourced Emotional Mutimodal Actors Dataset
- TESS: Toronto Emotional Speech Set
- RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
- EmoV-DB: Emotional Voices Database

The databases were not uploaded to Github due to copyright restrictions and size. However, my final processed database can be found [here](https://github.com/Brian-Pho/RVST598_Speech-Emotion-Recognition).

## Repo Structure and Description

```bash
└───repo  # Holds this repository
    ├───charts  # Holds pie charts created using Chart.js for the paper and presentation
    ├───data  # Holds the raw and processed databases (not uploaded to Github)
    │   ├───model  # Holds the neural network saved weights and training history
    │   ├───processed  # Holds the log-mel spectrogram audio samples in the form of numpy arrays
    │   └───raw  # Holds the raw audio samples and labels from each database
    │       ├───cremad
    │       ├───iemocap
    │       ├───ravdess
    │       └───tess
    ├───paper  # Holds the paper resulting from this research
    ├───presentation  # Holds the presentation resulting from this research
    └───src  # Holds the code for preprocessing the databases and the neural network model
        ├───audio_processor  # Holds the wav and spectrogram functions
        ├───database_processor  # Holds the preprocessing functions
        ├───neural_network  # Holds the neural network model
        ├───em_constants.py  # Holds the common emotions used throughout the program
        └───main.py  # Holds the main function to run to train and test the neural network model
```

## How to Run

I developed this program using PyCharm so that's the easiest way to run the code but its possible to run the code using a Terminal or Anaconda.

### Preprocess the Databases

1. Copy the raw database samples and labels into its respective folder in `/repo/data/raw/[db]`. The proper structure is below.
2. Run the corresponding database script in `/repo/src/database_processor/[db].py`
3. The processed log-mel spectrograms should appear in the `/repo/data/processed folder` with a filename like `CRE_0-0_1_0_0_0_0_0.npy`.

```bash
└───data  # Holds the raw and processed databases (not uploaded to Github)
    └───raw  # Holds the raw audio samples and labels from each database
        ├───cremad
        │   ├───tabulatedVotes.csv
        │   └───AudioWAV
        │       └───1001_DFA_ANG_XX.mp3
        ├───iemocap
        │   ├───data
        │   │   └───S1
        │   └───labels
        │       └───S1
        ├───ravdess
        │   └───Actor_01
        │       └───03-01-01-01-01-01-01.mp3
        └───tess
            └───data
                └───OAF_back_angry.mp3
```

### Train the Neural Network

1. Check the `/repo/data/processed` folder for files. If you're running the training on all four database, there should be 21,721 files/samples labeled like `CRE_0-0_1_0_0_0_0_0.npy`.
2. Configure the model parameters in `/repo/src/neural_network/nn_model.py` under the `build_model()` function.
3. Configure the training parameters under `/repo/src/neural_network/nn_constants.py`. This file lets you control the number of epochs, the batch size, the optimizer and loss, and more.
4. Run the training script `/repo/src/main.py`.
