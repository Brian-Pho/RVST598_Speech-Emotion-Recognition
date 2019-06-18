# RVST598_Speech-Emotion-Recognition

My summer research project on using machine learning to detect emotions in speech.

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

The following databases were used/considered

|            | Link                                                                   |       Using?       | File Types | Label Type | Label Level        |
|------------|------------------------------------------------------------------------|:------------------:|:----------:|------------|--------------------|
| IEMOCAP    | <https://sail.usc.edu/iemocap/>                                          | :heavy_check_mark: | wav        | Multi      | Sentence + Phoneme |
| MSP-IMPROV | <https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html> |         :x:        | ???        | ???        | ???                |
| CREMA-D    | <https://github.com/CheyneyComputerScience/CREMA-D>                      | :heavy_check_mark: | mp3 + wav  | Multi      |                    |
| TESS       | <https://tspace.library.utoronto.ca/handle/1807/24487>                   | :heavy_check_mark: | wav        | Single     |                    |
| RAVDESS    | <https://smartlaboratory.org/ravdess>                                    | :heavy_check_mark: | wav        | Single     | Sentence           |
| SAVEE      | <http://kahlan.eps.surrey.ac.uk/savee/Database.html>                     |         :x:        | wav        | Single     | Phoneme            |
| EmoV-DB    | <https://github.com/numediart/EmoV-DB>                                   |         :x:        | wav        | Single     |                    |

A table comparing the different emotions in each dataset.

|            |  Neutral |   Anger  |  Disgust |   Fear   |   Happy  |    Sad   | Surprise |   Calm   | Excitement | Frustration |  Amused  |  Sleepy  |
|------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------:|:----------:|:--------:|:--------:|
| IEMOCAP    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: |  :heavy_check_mark:  | :heavy_check_mark:   | :x: | :x: |
| MSP-IMPROV | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: |
| CREMA-D    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:   | :x:   | :x: | :x: |
| TESS       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: |
| RAVDESS    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:   | :x:   | :x: | :x: |
| SAVEE      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: |
| EmoV-DB    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x: | :x: | :x:   | :x:   | :heavy_check_mark: | :heavy_check_mark: |

The databases were not uploaded to Github due to their size and copy restrictions.
