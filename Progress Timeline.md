# Project Timeline

---

## Preparation

- [x] Define the research problem/question
- [x] Setup dev environment and tools
- [ ] Literature review
- [x] Find databases

## Main Work

- [x] Read in data from each database
- [x] Combine the different databases
- [x] Preprocess the data
- [x] Create the ML model
- [x] Train the ML model
- [ ] Modify the model to improve accuracy
- [ ] Complete experimentation
- [ ] Add title and axis labels for pictures

## Completion

- [ ] Write paper
- [ ] Create presentation
- [ ] Create poster
- [ ] Submit to conference

## Future work / Improvements

### Internal Improvements

- Improving accuracy, runtime, data efficiency, training efficiency, network efficiency
- More data, more data variety (actual conversations, movies, music), less noisy data
- Better multi-labels that output probabilities
- Use phase data from STFT
- Fuse spectrograms with raw waveforms using 1D CNN
- Combine with some RNN/LSTM
- Use MFCC
- Use neuroscience based approach/inspiration
- Use neural network for complex numbers
- Wavelets
- Pretrained network like soundnet
- Use different scales (log, Mel, custom, dB)
- Use different representation (wavelets, cepstrum)
- Use GAN for emotion style transfer
- Apply heatmaps to CNN + LSTM
- Use wavelet transform to deal with variable-length time signals
- More emotions
- Try binary relevance approach (each label is its own binary classifier)

### External Improvements

- Use this system as a component of a larger system
- Emotion style transfer using cycle-GANs
- Emotion generation using WavSynth/GAN/VAE/GANSynth
- Analyzing the neural network for more details on how it gets to its decision
- Use network for deaf people in a device
- Real-time emotion recognition

## TODO

- [x] Move normalization steps into frequency domain
- [x] Convert from normal frequency domain into mel-scaled
- [x] Save normalized, mel-scaled, freq domain samples
- [x] Split samples into train/valid/test
- [x] Calculate human baseline for emotion classification for IEMOCAP
- [x] Read in IEMOCAP
- [x] Update model to handle multi-label classification
- [x] Remove noise from Crema-d and Iemocap
- [x] Consider using EmoV-DB
- [x] Set the cutoff length for all database wav files
- [x] Calculate database statistics
- [x] Set new mel max hertz frequency cutoff as it's causing black bars on spectrograms
- [x] Update file_tree.txt
- [x] Use K-fold cross validation
