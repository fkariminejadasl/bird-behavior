Data
====

#### Unlabeled data
42,978,660 imu items (716311 data points 60 length or aka bursts or fixes), 3.1 GB (zip 458 MB) from CG_KREUPEL, LBBG_TEXEL, HG_TEXEL, CP_OMAN: From `226` GPS-timestamps files per device only `63` had 60 group length, based on my 10 attempt code (available_60points.csv). From 10 attempt GPS timestamps, if 1 returns I use the device. 

The code to get the data is in `scripts/get_data.py`.

#### Labeled data
The data is combination of several annotation. Since they are in different format and there was overlap between them, I had to make a separate data preparation script. It is in `scripts/prepare_labeled_data.py`.

Original data is 3505 fixes (70100 data points) of 20 length after removing the other class 3480 fixes​ remain. After combining data 8805 fixes (176100 data points) of 20 length and after remvong the other class 8742 fixes remain.
{0: 634, 1: 38, 2: 501, 3: 176, 4: 558, 5: 894, 6: 318, 7: 25, 8: 151, 9: 210}
{0: 1488, 1: 101, 2: 1142, 3: 352, 4: 1374, 5: 2676, 6: 755, 7: 63, 8: 335, 9: 519}
- final/s_data.csv # 3505 no others 3480
- final/combined_unique.csv # 8805 no others 8742
- data/combined_s_w_m_j.csv" # old 4394 no others 4365

Data preparation

1. Convert the original data to the CSV format
1. Get all data from database
1. Map to the closes divisilbe of 20 and get index from the database. Data is downloaded previously from databse using device id and dates (Previous step).
1. Combine data
1. Remove duplicates

Consideration
- The IMU and GPS are rouned and then saved to precision of 1e-6.
- It is possible to have the same IMU and GPS value. This issue is resolved by looking at two IMU and GPS​
- Data can start from the middle of burst (index not divisible by 20). This issue is resolved by mapping all data to divisible of 20.
- The s_data and possibly j_data are not sorted. 
- j_data, m_data: have different labels schema.​
- j_data, s_data: 10 and 20 items per burst, repectively. 
- m_data, w_data: data of different length.

Example row of the CSV format:
```bash
         0                    1  2  3         4         5         6         7
0      608  2013-05-31 02:12:41 -1  6 -1.020301 -0.305263  1.234586  0.186449
```


Data in memory
==============

#### CSV files

CSV files read by pandas are more efficient that reading py pure python or csv library. It is due to memory management such as contiguous block of memory and garbage collector, for-loop inefficiency of the pure python code and overhead due to dynamic typing and so on. For data above, only pandas version worked and used only 50% of memory wille pure python and csv reader killed by the process due to huge memory consumption, even using garbage collector `gc.collect()`. The experiment is in `panda_csv_python_read_data_in_memory.py`.

