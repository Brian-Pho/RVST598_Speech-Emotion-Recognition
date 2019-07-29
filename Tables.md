# Database Statistics

| Database | Number of Samples | Number of Outliers (> 3 STD) | Shortest | Longest | Shortest (after cutoff) | Longest (after cutoff) |
|----------|------------------:|-----------------------------:|---------:|--------:|------------------------:|-----------------------:|
| IEMOCAP  |             10039 |                              |          |         |                         |                        |
| CREMA-D  |              7442 |                           70 |    60861 |  240240 |                   60681 |                 193794 |
| TESS     |              2800 |                            0 |    60196 |  143271 |                   60196 |                 143271 |
| RAVDESS  |              1440 |                           16 |    44941 |  157053 |                   44941 |                 128224 |
| EmoV-DB  |              6599 |                              |          |         |                         |                        |

## Final Aggregate Database Statistics

Number of samples
Number of samples per emotion
Number of samples per database
Sample shape: (200, 278, 1)

## Accuracy and Agreement

| Database        | Accuracy (%) | Agreement (%) |
|-----------------|-------------:|--------------:|
| CREMA-D         | 61.89        | 68.77         |
| IEMOCAP         |              | 63.60         |
| MSP-Improv      | 69.40/73.30  |               |
| Yildirime et al | 68.30        |               |
