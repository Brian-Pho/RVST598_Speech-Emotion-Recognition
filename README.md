# RVST598_Speech-Emotion-Recognition

My summer research project on using machine learning to detect emotions in speech.

## Plan

My goal is to detect which emotions are present in speech. I consider these seven
emotions:

- Neutral
- Anger
- Disgust
- Fear
- Happy
- Sad
- Surprise

And have found databases with speech files spoken in these emotion. The data flow looks like this:

Raw waveform (time + amplitude) -> Spectrogram (time + frequency + ampltitude) -> Log-Mel Spectrogram (a spectrogram where frequency is converted into the mel scale) -> Convolutional Neural Network -> Output (one of the seven emotion)

## File structure

```bash
├───data
    ├───crema-d
    ├───iemocap
│   └───ravdess
├───src
│   └───database_formatter
```

## Database Information

The following databases were used/considered.

| Database                                                                             |       Using?       | File Type | Number of Files | Sampling Rate (Hz) | Label Type | Label Level        |
|--------------------------------------------------------------------------------------|:------------------:|:---------:|-----------------|--------------------|------------|--------------------|
| [IEMOCAP](https://sail.usc.edu/iemocap)                                              | :heavy_check_mark: |    wav    | 10039           | 16000              | Multi      | Sentence + Phoneme |
| [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html) |         :x:        |    ???    | ???             | ???                | ???        | ???                |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)                         | :heavy_check_mark: | mp3 + wav | 7442            | 16000              | Multi      | Sentence           |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)                         | :heavy_check_mark: |    wav    | 2800            | 24414              | Single     | Sentence           |
| [RAVDESS](https://smartlaboratory.org/ravdess)                                       | :heavy_check_mark: |    wav    | 1440            | 48000              | Single     | Sentence           |
| [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Database.html)                          |         :x:        |    wav    | 480             | 44100              | Single     | Sentence + Phoneme |
| [EmoV-DB](https://github.com/numediart/EmoV-DB)                                      |         :x:        |    wav    | 6599            | 44100              | Single     | Sentence           |
| [Emo-DB](http://emodb.bilderbar.info/index-1280.html)                                |         :x:        |    wav    | 535             | 16000              | Single     | Sentence           |

A table comparing the different emotions in each dataset.

|            |  Neutral |   Anger  |  Disgust |   Fear   |   Happy  |    Sad   | Surprise |   Calm   | Excitement | Frustration |  Amused  |  Sleepy  |  Bored  |
|------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------:|:----------:|:--------:|:--------:|:--------:|
| IEMOCAP    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: |  :heavy_check_mark:  | :heavy_check_mark:   | :x: | :x: | :x: |
| MSP-IMPROV | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| CREMA-D    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| TESS       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| RAVDESS    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:   | :x:   | :x: | :x: | :x: |
| SAVEE      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| EmoV-DB    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x: | :x: | :x:   | :x:   | :heavy_check_mark: | :heavy_check_mark: | :x: |
| Emo-DB     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:   | :x:   | :x: | :x: | :heavy_check_mark: 

The databases were not uploaded to Github due to their size and copy restrictions.

Database names

- IEMOCAP: Interactive Emotional Dyadic Motion Capture
- CREMA-D: Crowd-sourced Emotional Mutimodal Actors Dataset
- TESS: Toronto Emotional Speech Set
- RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
- EmoV-DB: Emotional Voices Database


## Random Notes

- Multiclass, multilabel problem
- Sentence-level emotion labeling
- Vocal emotion recognition, speech emotion recognition, emotion perception
- CREMA-D baseline 68.77 agreement
- Confusion matrix
- 