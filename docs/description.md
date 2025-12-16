## Data

### Unlabeled data

#### 20-length data (Gull only, more data)

The dataset includes `237` devices from `201` gulls, from which 15 devices doesn't have either IMU or calbration data, so in total `186` devices. It took `263` minutes to download the IMU and GPS data. GPS dates are downloaded quickly. `4.3GB` parquet data (`3.3GB` zip parquet, original `35GB` CSV data with `6.3 GB` zip in `15` min over `1075` files) contains `186` files, one for each device and `24,585,627` data point each with `20` items. Original data contains `25,543,410` data points before removing GPS 2D speed above `30 m/s`. The IMU data is clipped between -2 and 2 and GPS 2D speed is normalized by `22 m/s`.

```python
class_counts = 
'0: 3,896,949', '1: 297,034', '2: 2,845,657', '3: 268,561', '4: 1,863,161', '5: 11,622,311', '6: 1,361,568', '8: 758,359', '9: 1,676,835'
class_counts_percentage = 
{1: 0.01, 3: 0.01, 8: 0.03, 6: 0.06, 9: 0.07, 4: 0.08, 2: 0.12, 0: 0.16, 5: 0.47}
ind2name = 
{0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}
```

For example, device ID 6210 contains 1,562,699 GPS dates and 1,510,999 IMU and GPS data, each stored in sequences of length 20. This device has the largest total number of GPS dates.

The script first downloads the GPS dates, then downloads and filters the IMU and GPS data before saving them to files. For each device, the data is split into shards of 21,000 dates.

The code to generate the data is located in `scripts/get_data_gull_20.py`.

<details>
<summary>[Device: number of GPS dates. Click to expand]</summary>
no_imu = [45, 50, 51, 52, 53, 54, 130, 132, 135, 297, 326, 329]

nan_calibration = [121, 133, 134] also for [45, 50, 51, 52, 53, 54, 121, 130, 132, 133, 134, 135]

twice_in_ee_track_session_limited = [6009, 7001, 7021]

gull_devices = [45, 50, 51, 52, 53, 54, 121, 130, 132, 133, 134, 135, 297, 298, 304, 311, 317, 319, 320, 324, 325, 326, 327, 329, 344, 355, 373, 533, 534, 535, 536, 537, 538, 540, 541, 542, 604, 606, 608, 752, 754, 757, 781, 782, 784, 798, 805, 806, 868, 870, 871, 1600, 5387, 5388, 5390, 5391, 5392, 5393, 5415, 5416, 5472, 5473, 5496, 5586, 5587, 5588, 5590, 5591, 5592, 5593, 5594, 5599, 5601, 5603, 5687, 5689, 5690, 5692, 5693, 5694, 5697, 5698, 5699, 5700, 5701, 5702, 5704, 5705, 5992, 5993, 5995, 5996, 5997, 5998, 5999, 6004, 6006, 6009, 6011, 6012, 6014, 6015, 6016, 6017, 6071, 6072, 6073, 6074, 6075, 6076, 6077, 6079, 6080, 6082, 6202, 6205, 6206, 6208, 6210, 6212, 6214, 6216, 6217, 6219, 6387, 6392, 6395, 6396, 6397, 6399, 6400, 6401, 6402, 6403, 6501, 6502, 6503, 6504, 6506, 6507, 6508, 6509, 6510, 6511, 6512, 6513, 6514, 6515, 6516, 6517, 6518, 6519, 6520, 6521, 7001, 7002, 7004, 7007, 7008, 7009, 7021, 7036, 7045, 7046, 7047, 7048, 7050, 7055, 7056, 7058, 7059, 7061, 7062, 7063, 7077, 7078, 7079, 7080, 7081, 7083, 7085, 7086, 7120, 7123, 7126, 7131, 7132, 7133, 7135, 7137, 7138, 7139, 7140, 7142, 7145, 7147, 7148, 7149, 7160, 7161, 7162]

device_dates_counts = 
{45: 6835, 50: 964, 51: 18637, 52: 35364, 53: 6149, 54: 31670, 121: 1157, 130: 2091, 132: 1944, 133: 13828, 134: 7029, 135: 727, 297: 3775, 298: 5879, 304: 5607, 311: 35014, 317: 168310, 319: 1610, 320: 22565, 324: 2816, 325: 49112, 326: 5400, 327: 47999, 329: 42436, 344: 126109, 355: 25577, 373: 32871, 421: 18270, 446: 48857, 458: 3210, 533: 168583, 534: 253627, 535: 12524, 536: 11710, 537: 155853, 538: 14965, 540: 126638, 541: 75160, 542: 11624, 604: 9858, 606: 170982, 608: 65904, 640: 6428, 642: 1611, 644: 528, 646: 760, 657: 7567, 658: 8848, 659: 3251, 660: 128, 661: 932, 662: 23, 663: 1876, 670: 397, 671: 0, 672: 2412, 674: 25246, 675: 45738, 676: 5563, 680: 26947, 681: 20298, 682: 23444, 683: 18504, 688: 22403, 690: 13221, 752: 2066, 754: 205795, 757: 143468, 781: 41582, 782: 114749, 784: 2052, 798: 1490, 805: 313746, 806: 176918, 868: 5153, 870: 3774, 871: 238955, 1600: 176099, 2008: 6432, 2112: 7543, 2113: 1528, 2114: 4769, 2116: 7891, 2117: 54574, 2118: 31251, 2119: 21151, 2120: 339, 2121: 1638, 5387: 79123, 5388: 32062, 5390: 147831, 5391: 126108, 5392: 39381, 5393: 96670, 5415: 1433, 5416: 348618, 5472: 8262, 5473: 790625, 5496: 121257, 5586: 1404, 5587: 25, 5588: 197804, 5590: 3114, 5591: 37, 5592: 153421, 5593: 205762, 5594: 302808, 5599: 5220, 5601: 87504, 5603: 125455, 5687: 320396, 5689: 148628, 5690: 228, 5692: 35349, 5693: 460105, 5694: 227141, 5697: 83010, 5698: 86921, 5699: 46455, 5700: 80670, 5701: 23183, 5702: 52985, 5704: 235426, 5705: 116498, 5992: 6899, 5993: 32396, 5995: 188601, 5996: 73367, 5997: 111763, 5998: 145900, 5999: 47453, 6004: 47872, 6006: 435076, 6009: 72385, 6011: 147376, 6012: 34899, 6014: 9220, 6015: 320345, 6016: 331762, 6017: 373096, 6071: 15261, 6072: 128803, 6073: 689304, 6074: 146881, 6075: 5362, 6076: 24167, 6077: 52892, 6079: 50432, 6080: 361007, 6082: 2639, 6202: 213593, 6205: 88750, 6206: 332196, 6208: 91317, 6210: 1562699, 6212: 94176, 6214: 288608, 6216: 107960, 6217: 233357, 6219: 23914, 6387: 129084, 6392: 169844, 6395: 349709, 6396: 251334, 6397: 118675, 6399: 140538, 6400: 22169, 6401: 86693, 6402: 234476, 6403: 60984, 6501: 100562, 6502: 87082, 6503: 98214, 6504: 116412, 6506: 121193, 6507: 258215, 6508: 179569, 6509: 199277, 6510: 186051, 6511: 36190, 6512: 46811, 6513: 22874, 6514: 76022, 6515: 93600, 6516: 108067, 6517: 109611, 6518: 98174, 6519: 3062, 6520: 51335, 6521: 106584, 7001: 11215, 7002: 5239, 7004: 5225, 7007: 1277, 7008: 7126, 7009: 4819, 7021: 19181, 7036: 206719, 7045: 13362, 7046: 257212, 7047: 58469, 7048: 204935, 7050: 61365, 7055: 318181, 7056: 154, 7058: 40385, 7059: 109733, 7061: 220348, 7062: 168810, 7063: 69725, 7077: 4340, 7078: 34749, 7079: 2773, 7080: 4503, 7081: 3263, 7083: 5073, 7085: 1635, 7086: 34059, 7120: 1902, 7123: 897, 7126: 1435, 7131: 153063, 7132: 14299, 7133: 102538, 7135: 1259, 7137: 45824, 7138: 80940, 7139: 29756, 7140: 6120, 7142: 96598, 7145: 22470, 7147: 2096, 7148: 2031, 7149: 2177, 7160: 108954, 7161: 46255, 7162: 37384}
</details>

#### 60-length data (Gull and Crop Plover, less data)
42,978,660 IMU items (716311 data points of length 60, also known as bursts or fixes), 3.1 GB (zipped: 458 MB) from CG_KREUPEL, LBBG_TEXEL, HG_TEXEL, CP_OMAN: From `224` GPS timestamp files per device, only `63` had 60-group length, based on my 10-attempt code (`available_60points.csv`). From 10 GPS timestamp attempts, if 1 returns, I use the device. 

Of the 42,978,660 IMU items, 19,830,960 (CG_KREUPEL, LBBG_TEXEL, HG_TEXEL) are Gull data, and 23,147,700 are Crop Plover data (CP_OMAN).

The code to get the data is in `scripts/get_data_gull_cp_60.py`.

<details>
<summary>[Device: number of GPS dates. Click to expand]</summary>
{45: 6835, 50: 964, 51: 18637, 52: 35364, 53: 6149, 54: 31670, 121: 1157, 130: 2091, 132: 1944, 133: 13828, 134: 7029, 135: 727, 297: 3775, 298: 5879, 304: 5607, 311: 35014, 317: 168310, 319: 1610, 320: 22565, 324: 2816, 325: 49112, 326: 5400, 327: 47999, 329: 42436, 344: 126109, 355: 25577, 373: 32871, 421: 18270, 446: 48857, 458: 3210, 533: 168583, 534: 253627, 535: 12524, 536: 11710, 537: 155853, 538: 14965, 540: 126638, 541: 75160, 542: 11624, 604: 9858, 606: 170982, 608: 65904, 640: 6428, 642: 1611, 644: 528, 646: 760, 657: 7567, 658: 8848, 659: 3251, 660: 128, 661: 932, 662: 23, 663: 1876, 670: 397, 671: 0, 672: 2412, 674: 25246, 675: 45738, 676: 5563, 680: 26947, 681: 20298, 682: 23444, 683: 18504, 688: 22403, 690: 13221, 752: 2066, 754: 205795, 757: 143468, 781: 41582, 782: 114749, 784: 2052, 798: 1490, 805: 313746, 806: 176918, 868: 5153, 870: 3774, 871: 238955, 1600: 176099, 2008: 6432, 2112: 7543, 2113: 1528, 2114: 4769, 2116: 7891, 2117: 54574, 2118: 31251, 2119: 21151, 2120: 339, 2121: 1638, 5387: 79123, 5388: 32062, 5390: 147831, 5391: 126108, 5392: 39381, 5393: 96670, 5415: 1433, 5416: 348618, 5472: 8262, 5473: 790625, 5496: 121257, 5586: 1404, 5587: 25, 5588: 197804, 5590: 3114, 5591: 37, 5592: 153421, 5593: 205762, 5594: 302808, 5599: 5220, 5601: 87504, 5603: 125455, 5687: 320396, 5689: 148628, 5690: 228, 5692: 35349, 5693: 460105, 5694: 227141, 5697: 83010, 5698: 86921, 5699: 46455, 5700: 80670, 5701: 23183, 5702: 52985, 5704: 235426, 5705: 116498, 5992: 6899, 5993: 32396, 5995: 128710, 5996: 73367, 5998: 145900, 5999: 47453, 6004: 47872, 6006: 435076, 6009: 72385, 6011: 147376, 6012: 34899, 6014: 9220, 6015: 320345, 6016: 331762, 6017: 373096, 6071: 15261, 6072: 128803, 6073: 689304, 6074: 146881, 6075: 5362, 6076: 24167, 6077: 52892, 6079: 50432, 6080: 361007, 6082: 2639, 6202: 213593, 6205: 88750, 6206: 332196, 6208: 91317, 6212: 94176, 6214: 288608, 6216: 107960, 6217: 233357, 6219: 23914, 6387: 129084, 6392: 169844, 6395: 222193, 6396: 193394, 6397: 118675, 6400: 22169, 6401: 86693, 6402: 155329, 6403: 60984, 6501: 100562, 6502: 53395, 6503: 92919, 6504: 115164, 6506: 63602, 6507: 92191, 6508: 94375, 6509: 82911, 6510: 74843, 6511: 36190, 7001: 11215, 7002: 5239, 7004: 5225, 7007: 1277, 7008: 7126, 7009: 4819, 7021: 19181, 7036: 163355, 7045: 13362, 7046: 208593, 7047: 58469, 7048: 181939, 7050: 61365, 7055: 240955, 7056: 154, 7058: 40385, 7059: 83582, 7061: 151258, 7062: 117698, 7063: 69725, 7077: 4340, 7078: 6766, 7079: 2773, 7080: 4503, 7081: 3263, 7083: 5073, 7085: 1635, 7086: 34059, 7120: 1902, 7123: 897, 7126: 1435, 7131: 133660, 7132: 14299, 7133: 77046, 7135: 1259, 7137: 45824, 7138: 36336, 7139: 7034, 7140: 6120, 7142: 16013, 7145: 22470, 7147: 2096, 7148: 2031, 7149: 2177, 7160: 29507, 7161: 10593, 7162: 5596}
</details>



### Labeled data

#### Data Preparation Pipeline

The data is a combination of several datasets annotated by different experts. Each individual dataset is in a different format: JSON, MATLAB, or CSV. Some data appears in more than one dataset. We identified some mismatches between labels, which were either removed or corrected.

The data contains missing values and inconsistent label mappings. Since we have access to the database, missing items can be retrieved as needed.

The pipeline used to prepare the final dataset is implemented in `scripts/prepare_labeled_data.py`. It consists of two parts: one for individual datasets and one for the combined dataset.

{45: 6835, 50: 964, 51: 18637, 52: 35364, 53: 6149, 54: 31670, 121: 1157, 130: 2091, 132: 1944, 133: 13828, 134: 7029, 135: 727, 297: 3775, 298: 5879, 304: 5607, 311: 35014, 317: 168310, 319: 1610, 320: 22565, 324: 2816, 325: 49112, 326: 5400, 327: 47999, 329: 42436, 344: 126109, 355: 25577, 373: 32871, 421: 18270, 446: 48857, 458: 3210, 533: 168583, 534: 253627, 535: 12524, 536: 11710, 537: 155853, 538: 14965, 540: 126638, 541: 75160, 542: 11624, 604: 9858, 606: 170982, 608: 65904, 640: 6428, 642: 1611, 644: 528, 646: 760, 657: 7567, 658: 8848, 659: 3251, 660: 128, 661: 932, 662: 23, 663: 1876, 670: 397, 671: 0, 672: 2412, 674: 25246, 675: 45738, 676: 5563, 680: 26947, 681: 20298, 682: 23444, 683: 18504, 688: 22403, 690: 13221, 752: 2066, 754: 205795, 757: 143468, 781: 41582, 782: 114749, 784: 2052, 798: 1490, 805: 313746, 806: 176918, 868: 5153, 870: 3774, 871: 238955, 1600: 176099, 2008: 6432, 2112: 7543, 2113: 1528, 2114: 4769, 2116: 7891, 2117: 54574, 2118: 31251, 2119: 21151, 2120: 339, 2121: 1638, 5387: 79123, 5388: 32062, 5390: 147831, 5391: 126108, 5392: 39381, 5393: 96670, 5415: 1433, 5416: 348618, 5472: 8262, 5473: 790625, 5496: 121257, 5586: 1404, 5587: 25, 5588: 197804, 5590: 3114, 5591: 37, 5592: 153421, 5593: 205762, 5594: 302808, 5599: 5220, 5601: 87504, 5603: 125455, 5687: 320396, 5689: 148628, 5690: 228, 5692: 35349, 5693: 460105, 5694: 227141, 5697: 83010, 5698: 86921, 5699: 46455, 5700: 80670, 5701: 23183, 5702: 52985, 5704: 235426, 5705: 116498, 5992: 6899, 5993: 32396, 5995: 128710, 5996: 73367, 5998: 145900, 5999: 47453, 6004: 47872, 6006: 435076, 6009: 72385, 6011: 147376, 6012: 34899, 6014: 9220, 6015: 320345, 6016: 331762, 6017: 373096, 6071: 15261, 6072: 128803, 6073: 689304, 6074: 146881, 6075: 5362, 6076: 24167, 6077: 52892, 6079: 50432, 6080: 361007, 6082: 2639, 6202: 213593, 6205: 88750, 6206: 332196, 6208: 91317, 6212: 94176, 6214: 288608, 6216: 107960, 6217: 233357, 6219: 23914, 6387: 129084, 6392: 169844, 6395: 222193, 6396: 193394, 6397: 118675, 6400: 22169, 6401: 86693, 6402: 155329, 6403: 60984, 6501: 100562, 6502: 53395, 6503: 92919, 6504: 115164, 6506: 63602, 6507: 92191, 6508: 94375, 6509: 82911, 6510: 74843, 6511: 36190, 7001: 11215, 7002: 5239, 7004: 5225, 7007: 1277, 7008: 7126, 7009: 4819, 7021: 19181, 7036: 163355, 7045: 13362, 7046: 208593, 7047: 58469, 7048: 181939, 7050: 61365, 7055: 240955, 7056: 154, 7058: 40385, 7059: 83582, 7061: 151258, 7062: 117698, 7063: 69725, 7077: 4340, 7078: 6766, 7079: 2773, 7080: 4503, 7081: 3263, 7083: 5073, 7085: 1635, 7086: 34059, 7120: 1902, 7123: 897, 7126: 1435, 7131: 133660, 7132: 14299, 7133: 77046, 7135: 1259, 7137: 45824, 7138: 36336, 7139: 7034, 7140: 6120, 7142: 16013, 7145: 22470, 7147: 2096, 7148: 2031, 7149: 2177, 7160: 29507, 7161: 10593, 7162: 5596}



#### Individual Dataset Pipeline

Steps:

1. **format**: Convert data to CSV format with the following columns:

   * `device_id`, `datetimes`, `indices`, `label`, `IMU_x`, `IMU_y`, `IMU_z` (acceleration in g), and `GPS_2D_speed` (in m/s).
   * Missing labels or indices are represented by `-1`.
2. **index**: Add indices from the database.
3. **map0**: Apply a common, unified label mapping.
4. **mistakes**: Correct mismatched labels.
5. **map**: Merge some label classes and ignore others using `-1` as the label.
6. **drop negatives**: Remove entries where all labels are `-1`.
7. **complete**: Retrieve missing data (without labels) from the database.

#### Combined Dataset Pipeline

Steps:

1. **combine**: Merge all individual datasets into one.
2. **shift**: Group data into sets of 20 items. Assign a common label to each group or drop the group based on length and labeling rules. **starts** can be used instead. In this case, extracts a slice of n rows from the DataFrame starting at the first row where a valid label (0-9) appears.
3. **drop duplicates**: Perform a sanity check to ensure there are no duplicate groups of 20 items.


#### These datasets are:

```
s_data: (Publish data. set1 Json format)
j_data: (Suzanne Json)
m_data: (Suzanne Matlab)
w_data: (Willem Csv)
```

#### Considerations

- The IMU and GPS are rounded and then saved with a precision of 1e-6.
- It is possible to have the same IMU and GPS values. This issue is resolved by looking at three IMU and GPS items.
- Data can start from the middle of a burst (index not divisible by 20). Since we take bursts of 20 items, there are several ways to account for that. The rules are defined in `get_rules`. 
    - Shift each item by one and take 20 elements. This can also be implemented in data augmentation.
    - Mapping all data to values divisible by 20. (Old version in `exp/prepare_labeled_data`)
    - Other options: take the start of the signal (slice_from_first_label), a random part of the signal, or sample at a regular step.
- j_data, m_data have extra classes. The Float_groyne and Handling_mussel classes are mapped to Float and Pecking, respectively; the other classes are removed.
- j_data uses different ids for the same labels (remap: s_map0)
- j_data, s_data: 10 and 20 items per burst, respectively. 
- m_data, w_data: data of different lengths.

#### Here is the number of labels:

```bash
{0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}
# items
s_complete: {-1: 2422, 0: 12680, 1: 760, 2: 10020, 3: 3520, 4: 11160, 5: 17880, 6: 6280, 8: 2980, 9: 3740}
j_complete: {-1: 68059, 0: 2640, 1: 240, 2: 2310, 4: 5060, 5: 4170, 6: 1310, 8: 780, 9: 850}
m_complete: {-1: 5699, 0: 100, 4: 6587, 5: 12869, 6: 455, 9: 3327}
w_complete: {-1: 1040, 0: 13059, 1: 847, 2: 9977, 3: 3530, 4: 11230, 5: 16100, 6: 5842, 8: 2875, 9: 2907}
shift : {0: 161400, 1: 12040, 2: 113540, 3: 37760, 4: 168860, 5: 448760, 6: 86780, 8: 40700, 9: 92780}
# burst
s_index: {0: 634, 1: 38, 2: 501, 3: 176, 4: 558, 5: 894, 6: 318, 7: 25, 8: 151, 9: 210}
starts:  {0: 643, 1: 38, 2: 537, 3: 176, 4: 729, 5: 1502, 6: 337, 8: 151, 9: 225}
shift :  {0: 8070, 1: 602, 2: 5677, 3: 1888, 4: 8443, 5: 22438, 6: 4339, 8: 2035, 9: 4639}
```

#### Example row of the CSV format:

```bash
# no index (index==-1)
         0                    1  2  3         4         5         6         7
0      608  2013-05-31 02:12:41 -1  6 -1.020301 -0.305263  1.234586  0.186449
# with index
         0                    1  2  3         4         5         6         7
0      608  2013-05-31 02:12:41 20  6 -1.020301 -0.305263  1.234586  0.186449
```

### Previous Data preparation

#### There were some issues with the previous version:

    - Combining the items into bursts: We took the first label as the label of the burst. Now, priority labels are defined in get_rules.
    - Label StandForage used as Pecking. Now it is removed.
    - There was a bug in m_data. The labels were shifted if the items didn’t start from the beginning index. A maximum of 52 labels could have been wrong (test_issue_previous_m_data_format). Note that this is an upper bound. This is a minor issue.
    - Up to three items might have the same IMU and GPS speed. Previously, it was assumed only two. This is a very minor issue, since only a few items had this problem. It appears during reindexing by getting the index from the database.
    - The algorithm to find the closest value divisible by 20 had a minor issue with rounding. In Python, rounded values are to the closest even number. E.g., 2.5 is rounded to 2.0 instead of 3.0.

#### Description

The previous data preparation shifted the data to the closest value divisible by 20. So in the end, each data entry contains a burst of 20.

Data preparation is in `exp/prepare_labeled_data`. Here are the steps:

- Convert the original data to the CSV format
- Get all data from the database
- Map to the closest value divisible by 20 and get the index from the database. Data was previously downloaded using device ID and dates.
- Combine data (`combined.csv`)
- Remove duplicates (`combined_unique.csv`)
- Sort data by device and date in increasing index order (`combined_unique_sorted012.csv`).
- Manual correction by experts. The ground truth IMU figures were generated by `exps/save_plots_gt.py`. The correction is in the `corrections.txt` file and resulted in `corrected_combined_unique_sorted012.csv`.

> Balanced Data (`balance.csv`): For some experiments, I balanced the data by removing some rare classes and making the remaining classes have an equal number of instances based on the smallest class size. It is generated in data.py::balance_data by keeping {0: 'Flap', 1: 'Soar', 2: 'Float', 3: 'SitStand', 4: 'TerLoco', 5: 'Pecking'}.

In total, an extra 1235 bursts (c_data - s_data) were obtained. The "Other" class is excluded.

```
s_data: 3478  (Publish data. set1 Json format)
j_data: 1456  (Suzanne Json)
m_data: 857   (Suzanne Matlab)
w_data: 3341  (Willem Csv)
c_data: 4713  (Combined and duplicates removed without "other" class)
c_data: 4694  (Corrected c_data)
```

Here is the number of labels:

```
{0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}
s_data: {0: 634, 1: 38, 2: 501, 3: 176, 4: 558, 5: 894, 6: 318, 7: 25, 8: 151, 9: 210}
j_data: {0: 216, 1: 19, 2: 146, 3: 0,   4: 460, 5: 375,  6: 127, 7: 10, 8: 47,  9: 66}
m_data: {0: 5,   1: 0,  2: 0,   3: 0,   4: 0,   5: 642,  6: 23,  7: 0,  8: 0,   9: 187}
w_data: {0: 652, 1: 45, 2: 504, 3: 176, 4: 558, 5: 806,  6: 304, 7: 30, 8: 143, 9: 153}
```

### Data in memory

#### CSV files

CSV files read by pandas are more efficient that reading py pure python or csv library. It is due to memory management such as contiguous block of memory and garbage collector, for-loop inefficiency of the pure python code and overhead due to dynamic typing and so on. For data above, only pandas version worked and used only 50% of memory wille pure python and csv reader killed by the process due to huge memory consumption, even using garbage collector `gc.collect()`. The experiment is in `panda_csv_python_read_data_in_memory.py`.

#### Older method to generate labeled data

The older method is sandboxed in `sandbox_prepare_label_data` branch and in `scripts/prepare_labeled_data.py` file.

Older methods to generate data are:

- Method1: combine only based on device id and dates. This part is missing a lot of data.
- Method2: combine with device id, dates and labels. This part is has bug. 

These methods are slow since it request data directly from database. The method1 and method2 have the common part in getting data from the database. But the combined part differs. 

The combined part of the method 2 is the same as current method in `scripts/prepare_labeled_data.py`. 

> BUG of Method2 is in `write_j_data`, `write_m_data`: The data might be 0-19, 40-59 with label1 and 20-39 label2. Then 0-40 gets label1, and 20-40 label2. This caused because the maximum length is calculated based on common labels.

## Models

Number of parameters:

- Small Model: 6,309
- MaskedAutoencoderViT: 9,546,756
- TransformerEncoderMAE: 4,748,297

```python
from behavior import model as bm, model1d as bm1
model = bm.BirdModel(4, 30, 9)
model = bm1.MaskedAutoencoderViT(img_size=20, in_chans=4, patch_size=1, embed_dim=256, depth=6, num_heads=8, decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8, mlp_ratio=4, layer_norm_eps=1e-6)
model = bm1.TransformerEncoderMAE(img_size=20, in_chans=4, out_chans=9, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4, drop=0.0, layer_norm_eps=1e-6)
f"{sum([p.numel() for p in model.parameters()]):,}"
```

#### Small Bird Model

The bird model consists of three 1-D convolution layers, each with a kernel size of 3 and 30 channels. Each convolution layer is followed by batch normalization and a ReLU nonlinearity. The final projection layer is a linear layer that maps the average pooled embeddings to the number of classes. Cross-entropy loss is used as the classification loss. During training, dropout with a probability of 0.25 is applied after each convolution layer. The dataset for training consists of 3,505 labeled samples, each with 3 IMU accelerations and one GPS 2D speed for a sequence length of 20. There are 9 classes in total. AdamW is used for optimization with a learning rate of 3e-4 and a weight decay of 1e-2, with a maximum of 4,000 iterations. The learning rate scheduler is StepLR with a step size of 2,000, reducing the learning rate by a factor of 0.1. With this small dataset and model, the total training time is only 20 minutes using an NVIDIA GeForce RTX 3070 laptop GPU.


## List of scripts

#### App

- `app/bird_behavior_viz_app.py`: visualize IMU with behavior and the location in the map. The data for this app comes from `scripts/bird_behavior_viz.py`.
- `app/gps_locations_viz.py`: visualize GPS traces on the interactive map.

#### Visualization

- `scripts/embeddings_plot.py`: Plot embeddings (currently TSNE) given data and model.
- `scripts/bird_behavior_viz.py`: generate data (IMU, behavior classes by inference, GPS by downloading from database) and visualize IMU with behavior and location in the downloaded map. The visualization part is also in `app/bird_behavior_viz_app.py` with interactive mode.
- `exps/save_plots_gt.py`: save IMU plots for ground truth data based on device id and starting times.
- `exps/save_plots.py`: save IMU plots for each label, run inference for the predictions. 
- `generate_per_glen_figures_for_dt` used in `scripts/prepare_labeled_data.py`: generates IMU plots for structured data (divisible by glen=20) such as shift.csv. 
- `exps/imu_3d_movie`: make a movie from IMU data
- `exps/visualize`: old (I should remove)


#### Training/Inference Scripts

* `scripts/train.py`: Training script for supervised bird classification.
* `scripts/train_contrastive.py`: Similar to `scripts/train.py`, but includes additional losses (supervised contrastive loss and mean entropy maximization loss). The mean entropy maximization loss is currently not used due to lower performance.
* `scripts/ss_cluster_behavior.py`: Semi-supervised clustering script.
* `scripts/batch_train_cluster.py`: Runs batch experiments for training and clustering. Combines `scripts/train.py` and `scripts/ss_cluster_behavior.py` to execute multiple experiments simultaneously.
* `scripts/gcd_pipeline.py`: gcd pipeline
* `scripts/train_self_distill.py`: Training script for self-distillation. Currently, the infoNCE model is not trained due to imbalanced data (9 classes cause false negatives in the infoNCE loss). Instead, use DINO loss (positive pairs only) and change the augmentation to rotation of IMU data.

**Older scripts**: 

- `exps/cluster_behavior.py`: unsupervised clustering
- `exps/exps1`: Inference and save the metrics

#### Helper Scripts

- `scripts/prepare_labeled_data.py`: Script to run the pipeline in `data_processing.py`. See the description above.
- `scripts/get_data.py`: Script to get unlabeled data. See the description above.
- `exps/birdvis_query.py`: Generate query file to directly import in the birdvis tool.

#### Data 

- `create_five_balanced_data_and_unbalanced`: create five random balanced and unbalanced datasets from the given data file and save them to the specified path.

#### Analysis and Debugging

- `test_labels_comes_together`: check which labels appear together. This is obtained from `get_label_ranges` and `write_all_start_end_inds`, by calculating the index ranges in which the labels appear (device, time, label: start–end, ...).
- `identify_mistakes`: identify labeling mistakes. Find which data points are labeled differently. They are corrected in `correct_mistakes`, which is part of the data preparation pipeline.
- `test_find_index_jumps`: identify discontinuities in the indices. Determine whether the signal contains labeled segments separated by unlabeled intervals.
- `test_issue_previous_m_data_format`: check how impactful the bug is in the m_data format. The impact is not significant.
- `check_batches`: check that each batch with glen=20 meets certain conditions, such as consecutively increasing indices and unique values in columns 0, 1, 3, and 7 and having unique batches.
- `exps/panda_csv_python_read_data_in_memory`: check performance of panda, csv and pure python on reading large data in the memory 

## List of Functions

- `data.py::create_balanced_data`: Creates a balanced dataset by sampling equal number of groups from each specified class.
- `utils::stratified_split`: class-wise data split
- `map.py`: Visualize map from latitude longitude