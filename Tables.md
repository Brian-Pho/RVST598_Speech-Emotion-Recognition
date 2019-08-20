# Tables

## Databases

The following databases were used/considered.

| Database                                                                             | Year |       Using?       | File Type | Number of Files | Sampling Rate (Hz) | Label Type |     Label Level    |
|--------------------------------------------------------------------------------------|:----:|:------------------:|:---------:|:---------------:|:------------------:|:----------:|:------------------:|
| [IEMOCAP](https://sail.usc.edu/iemocap)                                              | 2008 | :heavy_check_mark: |    wav    |      10039      |        16000       |    Multi   | Sentence + Phoneme |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)                         | 2014 | :heavy_check_mark: | mp3 + wav |       7442      |        16000       |    Multi   |      Sentence      |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)                         | 2010 | :heavy_check_mark: |    wav    |       2800      |        24414       |   Single   |      Sentence      |
| [RAVDESS](https://smartlaboratory.org/ravdess)                                       | 2018 | :heavy_check_mark: |    wav    |       1440      |        48000       |   Single   |      Sentence      |
| [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html) | 2016 |         :x:        |    ???    |       ???       |         ???        |     ???    |         ???        |
| [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Database.html)                          | 2011 |         :x:        |    wav    |       480       |        44100       |   Single   | Sentence + Phoneme |
| [Emo-DB](http://emodb.bilderbar.info/index-1280.html)                                | 2005 |         :x:        |    wav    |       535       |        16000       |   Single   |      Sentence      |
| [EmoV-DB](https://github.com/numediart/EmoV-DB)                                      | 2018 |         :x:        |    wav    |       6599      |        44100       |   Single   |      Sentence      |

A comparision of the different emotions in each database.

|            |  Neutral |   Anger  |  Disgust |   Fear   |   Happy  |    Sad   | Surprise |   Calm   | Excitement | Frustration |  Amused  |  Sleepy  |  Bored  |
|------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:----------:|:----------:|:--------:|:--------:|:--------:|
| IEMOCAP    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: |  :heavy_check_mark:  | :heavy_check_mark:   | :x: | :x: | :x: |
| CREMA-D    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| TESS       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| RAVDESS    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:   | :x:   | :x: | :x: | :x: |
| MSP-IMPROV | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| SAVEE      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x:   | :x:   | :x: | :x: | :x: |
| Emo-DB     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:   | :x:   | :x: | :x: | :heavy_check_mark: |
| EmoV-DB    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x: | :x: | :x:   | :x:   | :heavy_check_mark: | :heavy_check_mark: | :x: |

## Final Aggregate Database

### Properties

| Property                       |         Value |
|--------------------------------|--------------:|
| Size                           |       4.50 GB |
| Size per Sample                |        218 KB |
| Number of Samples              |        21,675 |
| Sample Shape                   | (200, 278, 1) |

### Number of Samples per Emotion

| Emotion  | Number of Samples |
|----------|------------------:|
| Neutral  |              5954 |
| Anger    |              6177 |
| Disgust  |              1934 |
| Fear     |              1851 |
| Happy    |              4732 |
| Sad      |              3006 |
| Surprise |               876 |

### Number of Samples per Database

| Database | Number of Samples |
|----------|------------------:|
| IEMOCAP  |             10021 |
| CREMA-D  |              7414 |
| TESS     |              2800 |
| RAVDESS  |              1440 |

## Database Accuracy and Agreement

| Database        | Accuracy (%) | Agreement (%) |
|-----------------|-------------:|--------------:|
| CREMA-D         |        61.89 |         68.77 |
| IEMOCAP         |              |         63.60 |
| MSP-Improv      |  69.40/73.30 |               |
| Yildirime et al |        68.30 |               |

## Neural Network

| Statistic                      |      Value |
|--------------------------------|-----------:|
| Train Time                     |  1046.10 s |
| Number of Trainable Parameters | 76,584,615 |
| Test Loss                      |       0.88 |
| Test Accuracy                  |    45.36 % |

Tables were created with <https://www.tablesgenerator.com/markdown_tables>.
