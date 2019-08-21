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

### Architecture

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 198, 276, 200)     2000      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 99, 138, 200)      0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 99, 138, 200)      800       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 97, 136, 200)      360200    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 48, 68, 200)       0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 48, 68, 200)       800       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 46, 66, 150)       270150    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 33, 150)       0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 23, 33, 150)       600       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 21, 31, 100)       135100    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 10, 15, 100)       0         
_________________________________________________________________
batch_normalization_4 (Batch (None, 10, 15, 100)       400       
_________________________________________________________________
flatten_1 (Flatten)          (None, 15000)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               7680512   
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 512)               131584    
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 7)                 3591      
=================================================================
Total params: 8,717,065
Trainable params: 8,715,765
Non-trainable params: 1,300
_________________________________________________________________
```

### Training

| Statistic                      |      Value |
|--------------------------------|-----------:|
| Train Time                     |  2494.94 s |
| Number of Trainable Parameters |  8,715,765 |
| Test Loss                      |       0.36 |
| Test Accuracy                  |    52.16 % |
| Epochs                         |         20 |

Tables were created with <https://www.tablesgenerator.com/markdown_tables>.
