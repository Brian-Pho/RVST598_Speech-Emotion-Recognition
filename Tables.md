# Database Statistics

| Database | Number of Samples | Number of Outliers (> 3 STD) | Shortest | Longest | Shortest (after cutoff) | Longest (after cutoff) |
|----------|------------------:|-----------------------------:|---------:|--------:|------------------------:|-----------------------:|
| IEMOCAP  |             10039 |                              |          |         |                         |                        |
| CREMA-D  |              7442 |                           70 |    60861 |  240240 |                   60681 |                 193794 |
| TESS     |              2800 |                            0 |    60196 |  143271 |                   60196 |                 143271 |
| RAVDESS  |              1440 |                           16 |    44941 |  157053 |                   44941 |                 128224 |
| EmoV-DB  |              6599 |                              |          |         |                         |                        |

## Final Aggregate Database Statistics

| Statistic                      |         Value |
|--------------------------------|--------------:|
| Size                           |       4.50 GB |
| Size per Sample                |        218 KB |
| Number of Samples              |        21,721 |
| Sample Shape                   | (200, 278, 1) |
| Number of Samples per Emotion  |               |
| Number of Samples per Database |               |

## Accuracy and Agreement

| Database        | Accuracy (%) | Agreement (%) |
|-----------------|-------------:|--------------:|
| CREMA-D         |        61.89 |         68.77 |
| IEMOCAP         |              |         63.60 |
| MSP-Improv      |  69.40/73.30 |               |
| Yildirime et al |        68.30 |               |

## Neural Network Statistics

| Statistic                      |      Value |
|--------------------------------|-----------:|
| Train Time                     |  1046.10 s |
| Number of Trainable Parameters | 76,584,615 |
| Test Loss                      |       0.88 |
| Test Accuracy                  |    45.36 % |

Tables were created with <https://www.tablesgenerator.com/markdown_tables>.
