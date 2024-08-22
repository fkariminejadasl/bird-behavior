Data
====

42,978,660 imu items (716311 data points 60 length or aka bursts or fixes), 3.1 GB (zip 458 MB) from CG_KREUPEL, LBBG_TEXEL, HG_TEXEL, CP_OMAN: From `226` GPS-timestamps files per device only `63` had 60 group length, based on my 10 attempt code (available_60points.csv). From 10 attempt GPS timestamps, if 1 returns I use the device. 

The code to get the data is in `scripts/get_data.py`.

#### CSV files

CSV files read by pandas are more efficient that reading py pure python or csv library. It is due to memory management such as contiguous block of memory and garbage collector, for-loop inefficiency of the pure python code and overhead due to dynamic typing and so on. For data above, only pandas version worked and used only 50% of memory wille pure python and csv reader killed by the process due to huge memory consumption, even using garbage collector `gc.collect()`. The experiment is in `panda_csv_python_read_data_in_memory.py`.

