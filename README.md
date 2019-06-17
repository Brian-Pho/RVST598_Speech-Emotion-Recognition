# RVST598_Speech-Emotion-Recognition

My summer research project on using machine learning to detect emotions in speech.

## File structure

```bash
├───data
│   └───iemocap
│       ├───data
│       │   ├───S1
│       │   │   ├───Ses01F_impro01
│       │   │   ├───Ses01F_impro02
│       │   │   ├───...
│       │   ├───S2
│       │   │   ├───Ses02F_impro01
│       │   │   ├───Ses02F_impro02
│       │   │   ├───...
│       └───labels
│           ├───S1
│           ├───S2
│           ├───...
├───src
│   └───database_formatter
```

## Database Information

The following databases were used

- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
- [RAVDESS](https://smartlaboratory.org/ravdess)
- [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Database.html)
- [EmoV-DB](https://github.com/numediart/EmoV-DB)

A table comparing the different emotions in each dataset.

|            | Emotion |  Neutral |   Anger  |  Disgust |   Fear   |   Happy  |    Sad   | Surprise |   Calm   | Excitement | Fustration |
|------------|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------:|:----------:|
| Database   |         |          |          |          |          |          |          |          |          |            |            |
| IEMOCAP    |         | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |          |  &#10004;  | &#10004;   |
| MSP-IMPROV |         | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |          |            |            |
| CREMA-D    |         | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |          |          |            |            |
| TESS       |         | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |          |            |            |
| RAVDESS    |         | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |            |            |
| SAVEE      |         | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |          |            |            |
| EmoV-DB    |         |          |          |          |          |          |          |          |          |            |            |
|            |         |          |          |          |          |          |          |          |          |            |            |

The databases were not upload to Github due to their size and copy restrictions.
