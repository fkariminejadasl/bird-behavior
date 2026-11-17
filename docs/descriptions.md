# Description

Data, model, and script overview for the bird-behavior classifier. Companion
docs: [docs/status.md](status.md) (where the project stands, results tables),
[docs/experiments_log.md](experiments_log.md) (per-run notebook),
[docs/lesson_learned.md](lesson_learned.md) (curated lessons) and
[docs/methods.md](methods.md) (data rules, model and training).

# Quick reference

## Data

Labeled bursts: 20 samples at 20 Hz (1 s), 3 IMU axes + GPS speed. 9 classes.
Optionally 7 channels: `data.py::add_magnitude_features` appends `mag`,
`dyn_mag` and `jerk_mag`.

Terms (record, fix, burst, bout, segment): Data, Terms. The gull text says burst for a segment.

| lab | name      | paper      | starts   | %    | starts_clean | %    | dropped |
| --- | --------- | ---------- | -------- | ---- | ------------ | ---- | ------- |
| 0   | Flap      | flap       | 643      | 14.8 | 643          | 14.9 |         |
| 1   | ExFlap    | exflap     | 38       | 0.9  | 38           | 0.9  |         |
| 2   | Soar      | soar       | 537      | 12.4 | 537          | 12.5 |         |
| 3   | Boat      | boat       | 176      | 4.1  | 176          | 4.1  |         |
| 4   | Float     | float      | 729      | 16.8 | 729          | 16.9 |         |
| 5   | SitStand  | stationary | 1502     | 34.6 | 1492         | 34.6 | **10**  |
| 6   | TerLoco   | walk       | 337      | 7.8  | 326          | 7.6  | **11**  |
| 8   | Manouvre  | mixed      | 151      | 3.5  | 151          | 3.5  |         |
| 9   | Pecking   | peck       | 225      | 5.2  | 220          | 5.1  | **5**   |
|     | **total** |            | **4338** |      | **4312**     |      | **26**  |

`name` is the class name in this repo. `paper` is the name in "Shamoun-Baranes et al, 2016, Flap or soar? How a flight generalist responds to its aerial environment", in lower case: use it in published data. Label 7, Other, is removed.

Tags per study, and how `starts.csv` matches the published Movebank data: Labeled data, Tags and the published data.

- `scripts/data/prepare_labeled_data.py` — the whole labeled pipeline, ending in
  `starts.csv` then `starts_clean.csv`
- `exps/find_label_noise.py` — the 26 dropped bursts (GPS speed contradicts the label)
- `exps/label_cooccurrence.py` — which behaviours share a fix; 10 of 36 pairs
- `exps/inspect_unlabeled_data.py` — unlabeled data: curation and per-device distributions. Device lists: Data, Unlabeled data, Device lists
- How the labeled data was made before `starts.csv`: Data, Previous data preparation
- `behavior/data.py::acc_calibrate` — IMU calibration from the data, for a device without one
- `exps/spoonbill_data_stats.py` — spoonbill data (Lok et al.): 4718 segments, 13 behaviours, 8 birds; begging data: 436 segments, one bird. Spoonbill data, below
- `scripts/data/prepare_spoonbill_data.py` — the spoonbill training file, `spoonbill_starts.csv`: 5080 complete segments, 14 labels. Spoonbill data, Training file
- Spoonbill to gull label map, an ecologist's table: Spoonbill data, Spoonbill to gull labels

## Model

- `behavior/model.py::BirdModelSmallDilated` — `in_channels=4` **5,129 params**, `in_channels=7` **5,429 params**, receptive field 25
  (three k=5 convs, dilation 1/2/3), the app and baseline model
- `scripts/batch_train_supervised.py` — training; several configs in one run
- `scripts/train_joint.py` — one model on gulls and spoonbills, 18 joint classes, each label a set of them
- `behavior/data_augmentation.py` — per-sample `(T, C)` and batched `Batch*` `(N, C, T)`

## Eval

- `exps/eval_labeled.py` — accuracy, per-class F1 (plain + balanced), orientation
  robustness. Markdown tables for `status.md`
- `exps/eval_unlabeled.py` — no ground truth: impossible speed / place / flicker,
  and rotation instability
- `exps/speed_reliance.py` — how much a model relies on GPS speed: speed swept,
  shuffled or set to 0, acceleration kept
- `exps/eval_spoonbill.py` — spoonbill models against the two spoonbill
  papers: F1 per pooled behaviour, scored on records as the papers do, and on
  segments
- `exps/eval_cross_species.py` — each species' model on the other's data, scored by the label map
- `exps/eval_joint.py` — the joint models beside the single-species ones, on the same valid splits

Visual:

- `app/gps_burst_labeling_viz_app.py` — IMU trace + map + editable label
- `exps/save_plots_gt.py` — one PNG per labeled burst, in
  `/home/fatemeh/Downloads/bird/results/gt2_starts`
- `exps/plot_errors.py` — one PNG per valid burst a model gets wrong, in
  `/home/fatemeh/Downloads/bird/results/errors_exp<exp>`
- `behavior/utils.py::plot_one` — a single burst
- `scripts/embeddings_plot.py` — t-SNE of the learned features

# Data

### Terms

The spoonbill text uses these. The gull text is older: there a *burst* is a segment, and an *item* or *sample* is a record.

| term    | meaning                                                         | 60 records, one label |
| ------- | --------------------------------------------------------------- | --------------------- |
| record  | one row: one IMU reading x, y, z, with the GPS speed of its fix | 60                    |
| fix     | one GPS timestamp of one bird, with one position and speed      | 1                     |
| burst   | the records measured with one fix                               | 1                     |
| bout    | consecutive records of one fix with one label                   | 1                     |
| segment | 20 consecutive records with one label, the model input, 1 s     | 3                     |
| window  | 20 consecutive records from the start of a fix, any labels      | 3                     |

Lok et al. 2023 (Spoonbill data) also call the model input a segment, and a burst "a sample of 20 Hz tri-axial acceleration data". In machine learning *window* often means a segment; here it means only the cut on a fixed grid that the spoonbill papers score on (Spoonbill data, How the papers score). It can hold two behaviours, a segment cannot. *Sample* is avoided: in signal processing it means one reading.

### Unlabeled data

#### 20-length data (gulls only, more data)

- 237 devices from 201 gulls. 15 devices have no IMU or no calibration data,
  which leaves 186 devices, one Parquet file each.
- 24,585,627 bursts of 20 records, in
  `/home/fatemeh/Downloads/bird/data/ssl/ssl20parquet`. Before the 30 m/s cut
  there were 25,543,410. The IMU is clipped to [-2, 2] g and the GPS speed
  divided by 22.3 (`docs/methods.md`, rules 7-9).
- 4.3 GB of Parquet (3.3 GB zipped), from 35 GB of CSV (6.3 GB zipped) in 1075
  files. Downloading the IMU and GPS took 263 minutes; the GPS timestamps
  download quickly.
- The largest device, 6210, has 1,562,699 GPS timestamps and 1,510,999 bursts.
- Made in two steps. `scripts/data/get_data_gull_20.py` downloads the GPS
  timestamps, then the IMU and GPS, in CSV shards of 21,000 timestamps per
  device. `exps/inspect_unlabeled_data.py` curates the shards (the cut, the
  clip and the division above), writes one Parquet file per device, and plots
  the per-device distributions that motivate the curation.

#### 60-length data (gulls and crab plovers, less data)

- 716,311 bursts of 60 records (42,978,660 records), 3.1 GB (458 MB zipped),
  from CG_KREUPEL, LBBG_TEXEL, HG_TEXEL and CP_OMAN. The three gull studies
  hold 19,830,960 records, the crab plovers (CP_OMAN) 23,147,700.
- Of the 224 GPS timestamp files, one per device, 63 have bursts of 60 records
  (`available_60points.csv`). A device is used if at least one of 10 download
  attempts returns data.
- Made by `scripts/data/get_data_gull_cp_60.py`.

#### Device lists

Kept here to look up which devices exist. Click to expand.

<details>
<summary>20-length data: devices without IMU or calibration, all gull devices, GPS timestamps per device</summary>
no_imu = [45, 50, 51, 52, 53, 54, 130, 132, 135, 297, 326, 329]

nan_calibration = [121, 133, 134] also for [45, 50, 51, 52, 53, 54, 121, 130, 132, 133, 134, 135]

twice_in_ee_track_session_limited = [6009, 7001, 7021]

gull_devices = [45, 50, 51, 52, 53, 54, 121, 130, 132, 133, 134, 135, 297, 298, 304, 311, 317, 319, 320, 324, 325, 326, 327, 329, 344, 355, 373, 533, 534, 535, 536, 537, 538, 540, 541, 542, 604, 606, 608, 752, 754, 757, 781, 782, 784, 798, 805, 806, 868, 870, 871, 1600, 5387, 5388, 5390, 5391, 5392, 5393, 5415, 5416, 5472, 5473, 5496, 5586, 5587, 5588, 5590, 5591, 5592, 5593, 5594, 5599, 5601, 5603, 5687, 5689, 5690, 5692, 5693, 5694, 5697, 5698, 5699, 5700, 5701, 5702, 5704, 5705, 5992, 5993, 5995, 5996, 5997, 5998, 5999, 6004, 6006, 6009, 6011, 6012, 6014, 6015, 6016, 6017, 6071, 6072, 6073, 6074, 6075, 6076, 6077, 6079, 6080, 6082, 6202, 6205, 6206, 6208, 6210, 6212, 6214, 6216, 6217, 6219, 6387, 6392, 6395, 6396, 6397, 6399, 6400, 6401, 6402, 6403, 6501, 6502, 6503, 6504, 6506, 6507, 6508, 6509, 6510, 6511, 6512, 6513, 6514, 6515, 6516, 6517, 6518, 6519, 6520, 6521, 7001, 7002, 7004, 7007, 7008, 7009, 7021, 7036, 7045, 7046, 7047, 7048, 7050, 7055, 7056, 7058, 7059, 7061, 7062, 7063, 7077, 7078, 7079, 7080, 7081, 7083, 7085, 7086, 7120, 7123, 7126, 7131, 7132, 7133, 7135, 7137, 7138, 7139, 7140, 7142, 7145, 7147, 7148, 7149, 7160, 7161, 7162]

device_dates_counts =
{45: 6835, 50: 964, 51: 18637, 52: 35364, 53: 6149, 54: 31670, 121: 1157, 130: 2091, 132: 1944, 133: 13828, 134: 7029, 135: 727, 297: 3775, 298: 5879, 304: 5607, 311: 35014, 317: 168310, 319: 1610, 320: 22565, 324: 2816, 325: 49112, 326: 5400, 327: 47999, 329: 42436, 344: 126109, 355: 25577, 373: 32871, 421: 18270, 446: 48857, 458: 3210, 533: 168583, 534: 253627, 535: 12524, 536: 11710, 537: 155853, 538: 14965, 540: 126638, 541: 75160, 542: 11624, 604: 9858, 606: 170982, 608: 65904, 640: 6428, 642: 1611, 644: 528, 646: 760, 657: 7567, 658: 8848, 659: 3251, 660: 128, 661: 932, 662: 23, 663: 1876, 670: 397, 671: 0, 672: 2412, 674: 25246, 675: 45738, 676: 5563, 680: 26947, 681: 20298, 682: 23444, 683: 18504, 688: 22403, 690: 13221, 752: 2066, 754: 205795, 757: 143468, 781: 41582, 782: 114749, 784: 2052, 798: 1490, 805: 313746, 806: 176918, 868: 5153, 870: 3774, 871: 238955, 1600: 176099, 2008: 6432, 2112: 7543, 2113: 1528, 2114: 4769, 2116: 7891, 2117: 54574, 2118: 31251, 2119: 21151, 2120: 339, 2121: 1638, 5387: 79123, 5388: 32062, 5390: 147831, 5391: 126108, 5392: 39381, 5393: 96670, 5415: 1433, 5416: 348618, 5472: 8262, 5473: 790625, 5496: 121257, 5586: 1404, 5587: 25, 5588: 197804, 5590: 3114, 5591: 37, 5592: 153421, 5593: 205762, 5594: 302808, 5599: 5220, 5601: 87504, 5603: 125455, 5687: 320396, 5689: 148628, 5690: 228, 5692: 35349, 5693: 460105, 5694: 227141, 5697: 83010, 5698: 86921, 5699: 46455, 5700: 80670, 5701: 23183, 5702: 52985, 5704: 235426, 5705: 116498, 5992: 6899, 5993: 32396, 5995: 188601, 5996: 73367, 5997: 111763, 5998: 145900, 5999: 47453, 6004: 47872, 6006: 435076, 6009: 72385, 6011: 147376, 6012: 34899, 6014: 9220, 6015: 320345, 6016: 331762, 6017: 373096, 6071: 15261, 6072: 128803, 6073: 689304, 6074: 146881, 6075: 5362, 6076: 24167, 6077: 52892, 6079: 50432, 6080: 361007, 6082: 2639, 6202: 213593, 6205: 88750, 6206: 332196, 6208: 91317, 6210: 1562699, 6212: 94176, 6214: 288608, 6216: 107960, 6217: 233357, 6219: 23914, 6387: 129084, 6392: 169844, 6395: 349709, 6396: 251334, 6397: 118675, 6399: 140538, 6400: 22169, 6401: 86693, 6402: 234476, 6403: 60984, 6501: 100562, 6502: 87082, 6503: 98214, 6504: 116412, 6506: 121193, 6507: 258215, 6508: 179569, 6509: 199277, 6510: 186051, 6511: 36190, 6512: 46811, 6513: 22874, 6514: 76022, 6515: 93600, 6516: 108067, 6517: 109611, 6518: 98174, 6519: 3062, 6520: 51335, 6521: 106584, 7001: 11215, 7002: 5239, 7004: 5225, 7007: 1277, 7008: 7126, 7009: 4819, 7021: 19181, 7036: 206719, 7045: 13362, 7046: 257212, 7047: 58469, 7048: 204935, 7050: 61365, 7055: 318181, 7056: 154, 7058: 40385, 7059: 109733, 7061: 220348, 7062: 168810, 7063: 69725, 7077: 4340, 7078: 34749, 7079: 2773, 7080: 4503, 7081: 3263, 7083: 5073, 7085: 1635, 7086: 34059, 7120: 1902, 7123: 897, 7126: 1435, 7131: 153063, 7132: 14299, 7133: 102538, 7135: 1259, 7137: 45824, 7138: 80940, 7139: 29756, 7140: 6120, 7142: 96598, 7145: 22470, 7147: 2096, 7148: 2031, 7149: 2177, 7160: 108954, 7161: 46255, 7162: 37384}

</details>

<details>
<summary>60-length data: GPS timestamps per device</summary>
{45: 6835, 50: 964, 51: 18637, 52: 35364, 53: 6149, 54: 31670, 121: 1157, 130: 2091, 132: 1944, 133: 13828, 134: 7029, 135: 727, 297: 3775, 298: 5879, 304: 5607, 311: 35014, 317: 168310, 319: 1610, 320: 22565, 324: 2816, 325: 49112, 326: 5400, 327: 47999, 329: 42436, 344: 126109, 355: 25577, 373: 32871, 421: 18270, 446: 48857, 458: 3210, 533: 168583, 534: 253627, 535: 12524, 536: 11710, 537: 155853, 538: 14965, 540: 126638, 541: 75160, 542: 11624, 604: 9858, 606: 170982, 608: 65904, 640: 6428, 642: 1611, 644: 528, 646: 760, 657: 7567, 658: 8848, 659: 3251, 660: 128, 661: 932, 662: 23, 663: 1876, 670: 397, 671: 0, 672: 2412, 674: 25246, 675: 45738, 676: 5563, 680: 26947, 681: 20298, 682: 23444, 683: 18504, 688: 22403, 690: 13221, 752: 2066, 754: 205795, 757: 143468, 781: 41582, 782: 114749, 784: 2052, 798: 1490, 805: 313746, 806: 176918, 868: 5153, 870: 3774, 871: 238955, 1600: 176099, 2008: 6432, 2112: 7543, 2113: 1528, 2114: 4769, 2116: 7891, 2117: 54574, 2118: 31251, 2119: 21151, 2120: 339, 2121: 1638, 5387: 79123, 5388: 32062, 5390: 147831, 5391: 126108, 5392: 39381, 5393: 96670, 5415: 1433, 5416: 348618, 5472: 8262, 5473: 790625, 5496: 121257, 5586: 1404, 5587: 25, 5588: 197804, 5590: 3114, 5591: 37, 5592: 153421, 5593: 205762, 5594: 302808, 5599: 5220, 5601: 87504, 5603: 125455, 5687: 320396, 5689: 148628, 5690: 228, 5692: 35349, 5693: 460105, 5694: 227141, 5697: 83010, 5698: 86921, 5699: 46455, 5700: 80670, 5701: 23183, 5702: 52985, 5704: 235426, 5705: 116498, 5992: 6899, 5993: 32396, 5995: 128710, 5996: 73367, 5998: 145900, 5999: 47453, 6004: 47872, 6006: 435076, 6009: 72385, 6011: 147376, 6012: 34899, 6014: 9220, 6015: 320345, 6016: 331762, 6017: 373096, 6071: 15261, 6072: 128803, 6073: 689304, 6074: 146881, 6075: 5362, 6076: 24167, 6077: 52892, 6079: 50432, 6080: 361007, 6082: 2639, 6202: 213593, 6205: 88750, 6206: 332196, 6208: 91317, 6212: 94176, 6214: 288608, 6216: 107960, 6217: 233357, 6219: 23914, 6387: 129084, 6392: 169844, 6395: 222193, 6396: 193394, 6397: 118675, 6400: 22169, 6401: 86693, 6402: 155329, 6403: 60984, 6501: 100562, 6502: 53395, 6503: 92919, 6504: 115164, 6506: 63602, 6507: 92191, 6508: 94375, 6509: 82911, 6510: 74843, 6511: 36190, 7001: 11215, 7002: 5239, 7004: 5225, 7007: 1277, 7008: 7126, 7009: 4819, 7021: 19181, 7036: 163355, 7045: 13362, 7046: 208593, 7047: 58469, 7048: 181939, 7050: 61365, 7055: 240955, 7056: 154, 7058: 40385, 7059: 83582, 7061: 151258, 7062: 117698, 7063: 69725, 7077: 4340, 7078: 6766, 7079: 2773, 7080: 4503, 7081: 3263, 7083: 5073, 7085: 1635, 7086: 34059, 7120: 1902, 7123: 897, 7126: 1435, 7131: 133660, 7132: 14299, 7133: 77046, 7135: 1259, 7137: 45824, 7138: 36336, 7139: 7034, 7140: 6120, 7142: 16013, 7145: 22470, 7147: 2096, 7148: 2031, 7149: 2177, 7160: 29507, 7161: 10593, 7162: 5596}
</details>

#### Reading large CSV files

Reading the CSV shards with pandas works where pure Python and the `csv` module
run out of memory: on the 20-length data, pandas used 50% of the memory, and the
other two were killed even with `gc.collect()`
(`exps/panda_csv_python_read_data_in_memory.py`). Notes on loading the Parquet
files for training: `docs/cheatsheets/efficient_data_loading.md`.

### Labeled data

#### Data Preparation Pipeline

The data is a combination of several datasets annotated by different experts. Each individual dataset is in a different format: JSON, MATLAB, or CSV. Some data appears in more than one dataset. We identified some mismatches between labels, which were either removed or corrected.

The data contains missing values and inconsistent label mappings. Since we have access to the database, missing items can be retrieved as needed.

The pipeline used to prepare the final dataset is implemented in `scripts/data/prepare_labeled_data.py`. It consists of two parts: one for individual datasets and one for the combined dataset.

#### Individual Dataset Pipeline

Steps:

1. **format**: Convert data to CSV format with the following columns:

   - `device_id`, `datetimes`, `indices`, `label`, `IMU_x`, `IMU_y`, `IMU_z` (acceleration in g), and `GPS_2D_speed` (in m/s).
   - Missing labels or indices are represented by `-1`.

2. **index**: Add indices from the database.

3. **map0**: Apply a common, unified label mapping.

4. **mistakes**: Correct mismatched labels.

5. **map**: Merge some label classes and ignore others using `-1` as the label.

6. **drop negatives**: Remove entries where all labels are `-1`.

7. **complete**: Retrieve missing data (without labels) from the database.

#### Combined Dataset Pipeline

Steps:

1. **combine**: Merge all individual datasets into one.
2. **shift**: Group data into sets of 20 items. Assign a common label to each group or drop the group based on length and labeling rules. **starts** can be used instead. In this case, it extracts a slice of n rows from the DataFrame, starting at the first row where a valid label (0-9) appears.
3. **drop duplicates**: Perform a sanity check to ensure there are no duplicate groups of 20 items.
4. **remove label noise**: Drop the 26 bursts whose GPS speed contradicts their label, found by `exps/find_label_noise.py` and hard-coded in `data_processing.remove_label_noise`. Reads `starts.csv` and writes `starts_clean.csv` (4312 bursts). This runs after `make_combined_data_pipeline` and can be rerun on its own, without the database steps above.

#### These datasets are:

```
s_data: (published data, set1 JSON format)
j_data: (Suzanne JSON)
m_data: (Suzanne MATLAB)
w_data: (Willem CSV)
```

#### Considerations

- The IMU and GPS are rounded and then saved with a precision of 1e-6.
- It is possible to have the same IMU and GPS values. This issue is resolved by looking at three consecutive IMU and GPS items.
- Data can start from the middle of a burst (index not divisible by 20). Since we take bursts of 20 items, there are several ways to account for that. The rules are defined in `get_rules`.
  - Shift each item by one and take 20 elements. This can also be implemented as data augmentation.
  - Map all data to values divisible by 20. (Old version in `exps/prepare_labeled_data.py`)
  - Other options: take the start of the signal (`slice_from_first_label`), a random part of the signal, or samples at regular intervals.
- j_data and m_data have extra classes. The Float_groyne and Handling_mussel classes are mapped to Float and Pecking, respectively; the other classes are removed.
- j_data uses different IDs for the same labels (remap: s_map0).
- j_data and s_data have 10 and 20 items per burst, respectively.
- m_data, w_data: data of different lengths.

#### Counts per label at each step

Labels as in the Quick reference table. `starts` and `starts_clean`: that table.

```bash
# records
s_complete: {-1: 2422, 0: 12680, 1: 760, 2: 10020, 3: 3520, 4: 11160, 5: 17880, 6: 6280, 8: 2980, 9: 3740}
j_complete: {-1: 68059, 0: 2640, 1: 240, 2: 2310, 4: 5060, 5: 4170, 6: 1310, 8: 780, 9: 850}
m_complete: {-1: 5699, 0: 100, 4: 6587, 5: 12869, 6: 455, 9: 3327}
w_complete: {-1: 1040, 0: 13059, 1: 847, 2: 9977, 3: 3530, 4: 11230, 5: 16100, 6: 5842, 8: 2875, 9: 2907}
shift : {0: 161400, 1: 12040, 2: 113540, 3: 37760, 4: 168860, 5: 448760, 6: 86780, 8: 40700, 9: 92780}
# bursts of 20 records
s_index: {0: 634, 1: 38, 2: 501, 3: 176, 4: 558, 5: 894, 6: 318, 7: 25, 8: 151, 9: 210}
shift :  {0: 8070, 1: 602, 2: 5677, 3: 1888, 4: 8443, 5: 22438, 6: 4339, 8: 2035, 9: 4639}
```

#### Behaviour groups (old notes)

Counts are `s_data` bursts.

```
Flight:  0 Flap 634, 1 ExFlap 38, 2 Soar 501 (thermal soaring), 8 Manoeuvre/Mixed 151
Float:   4 Float 558
SitStand:5 SitStand 894, 3 Boat 176
TerLoco: 6 TerLoco/Walk 318, 9 Pecking 210 (paper 209)
Other:   7 Other 25
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

#### Tags and the published data

The 19 gulls in `starts.csv` are in two studies published on Movebank and Zenodo. Checked against each study's reference data on Zenodo (`HG_TEXEL-reference-data.csv`, `LBBG_TEXEL-reference-data.csv`): each tag has one deployment, and all labeled times fall inside it.

- HG_TEXEL ([10.5281/zenodo.20706394](https://doi.org/10.5281/zenodo.20706394)), herring gull (*Larus argentatus*), 5 tags: 1600, 6011, 6016, 6080, 6210.
- LBBG_TEXEL ([10.5281/zenodo.20706372](https://doi.org/10.5281/zenodo.20706372)), lesser black-backed gull (*Larus fuscus*), 14 tags: 533, 534, 537, 541, 606, 608, 754, 757, 781, 782, 798, 805, 806, 871.

The values equal the published ones, up to the 1e-6 rounding. `IMU_x`, `IMU_y`, `IMU_z` are `tilt-x`, `tilt-y`, `tilt-z`, and `GPS_2D_speed` is `ground-speed`. `datetimes` is the GPS `timestamp` and the acceleration `start-timestamp`; the sample with index n is at `start-timestamp` + n × 50 ms. This was a one-off check on tag 6210 (all 2480 of its labeled samples, against the HG_TEXEL 2016 files); there is no script for it.

### Spoonbill data

Eurasian spoonbill data, in `/home/fatemeh/Downloads/bird/data/spoonbill`. Terms as in Terms, above.

- Paper: Lok et al. 2023, Prey ingestion rates revealed by back-mounted accelerometers in Eurasian spoonbills, Animal Biotelemetry 11:5, [10.1186/s40317-022-00315-w](https://doi.org/10.1186/s40317-022-00315-w). Data: [doi.nioz.nl/dataset/7bbyd](https://doi.nioz.nl/dataset/7bbyd), with 10 videos left out here. Code: [zenodo.org/records/7541131](https://zenodo.org/records/7541131).
- A second paper adds begging: Begging data, below.

One row per record, as for the gulls: IMU `x`, `y`, `z` in g and GPS `speed_2d`, one speed per fix, one burst per fix. A fix is not 20 records: in the video file it holds up to 200 (10 s), so up to 10 segments.

20 Hz, as in the paper. Checked on the video clock `time.on.video`, one value per second: count the records of a fix that share one second. The first and last second of a fix are cut off by the burst start and end, so they are skipped. All 4376 seconds of video left hold exactly 20 records (`exps/spoonbill_data_stats.py`). 4376 counts seconds of video for this check only, not segments.

| file                                       | what                               | birds                    | fixes | records per fix  |
| ------------------------------------------ | ---------------------------------- | ------------------------ | ----- | ---------------- |
| `video_annotated_data.csv`                 | labeled from video, one per record | 584, 656, 760, 763, 1608 | 690   | median 200, 10 s |
| `visual_annotated_passive_flight_data.csv` | fly-soar only, labeled by eye      | 6283, 6287, 6291         | 51    | 32               |
| `acc_data_migration_<bird>.csv`            | migration, no labels               | 6283, 6287, 6291         | 3872  | mostly 32        |
| `behaviours.csv`                           | label index to name, 16 labels     |                          |       |                  |

The videos hold "very few data on passive (soaring/gliding) flight", so the paper labeled it from plots of the migration data of three birds. The soaring file is that labeled copy of 51 migration fixes: same rows, same values. A soaring fix holds 32 records, so it gives one segment: 51 fixes, 51 segments. The migration files give 3694 segments, unlabeled and not used for training; the 51 soaring ones are among them.

Segments of 20 records, cut as `data_processing.slice_from_first_label` cuts `starts.csv`: a bout of n records gives n // 20 segments. The table is the `Segments of 20 records per behaviour` output of `exps/spoonbill_data_stats.py`; `main bird` comes from its `Segments per behaviour and bird` output.

- `records`, `bouts`, `segments`: counts per behaviour.
- `bouts < 20`: bouts too short for one segment. All their records are lost.
- `% of segments`: this behaviour's segments out of all 4718.
- `% records used`: this behaviour's records that land in a segment. Records are lost in short bouts and at the end of longer ones: a 35-record bout gives 1 segment and leaves 15.
- `complete`: segments with no missing x, y, z or speed.
- `main bird`: the bird with the most segments of this behaviour, and its share.

| lab | name                 | records    | bouts    | bouts < 20 | segments | % of segments | % records used | complete | main bird  |
| --- | -------------------- | ---------- | -------- | ---------- | -------- | ------------- | -------------- | -------- | ---------- |
| 2   | stand-rest           | 29540      | 200      | 1          | 1415     | 30.0          | 96             | 1357     | 763 (94%)  |
| 4   | stand-preen          | 6959       | 96       | 13         | 305      | 6.5           | 88             | 295      | 763 (91%)  |
| 6   | sit-alert            | 1249       | 8        | 0          | 62       | 1.3           | 99             | 62       | 584 (100%) |
| 7   | stand-alert          | 9529       | 138      | 26         | 416      | 8.8           | 87             | 416      | 763 (61%)  |
| 8   | fly-flap             | 3794       | 27       | 1          | 185      | 3.9           | 98             | 185      | 760 (50%)  |
| 9   | fly-soar             | 1722       | 56       | 4          | 53       | 1.1           | 62             | 52       | 6283 (42%) |
| 10  | walk                 | 4574       | 101      | 36         | 177      | 3.8           | 77             | 177      | 760 (69%)  |
| 11  | for-search           | 47264      | 933      | 267        | 1935     | 41.0          | 82             | 1929     | 760 (90%)  |
| 12  | for-handle           | 797        | 42       | 31         | 14       | 0.3           | 35             | 14       | 760 (93%)  |
| 13  | for-intake           | 9459       | 571      | 456        | 115      | 2.4           | 24             | 115      | 760 (98%)  |
| 14  | stand-other          | 566        | 21       | 9          | 16       | 0.3           | 57             | 15       | 763 (100%) |
| 15  | stand-shake-feathers | 559        | 23       | 13         | 18       | 0.4           | 64             | 18       | 763 (72%)  |
| 16  | drink                | 173        | 4        | 0          | 7        | 0.1           | 81             | 7        | 763 (100%) |
|     | **total**            | **116185** | **2220** | **857**    | **4718** | **100**       | **81**         | **4642** |            |

- sit-rest, sit-preen and washing (1, 3, 5) have no records.
- fly-soar: 53 segments, 2 from video and 51 from the soaring file; 52 complete.
- Stride-1 segments inside one bout, like the gull `shift`, give 78,842.
- 5 records of fix 760, 2014-06-08 12:19:53 carry two labels (for-search and for-intake). Both copies are dropped.
- 12 video fixes have no speed. Speed is one value per fix, so all their records lack it: 53 to 200 records per fix, 1620 in all, 73 segments. Most of the 76 incomplete segments are these.
- 41 video records lack z, 20 of them also y. 248 are beyond 2 g, the gull clip.

#### Begging data

Begging is a chick walking after a parent and asking to be fed.

- Paper: Lok et al. 2025, Eurasian Spoonbill chicks receive parental care up to several months after fledging, but not into migration, [10.1007/s00265-025-03649-6](https://doi.org/10.1007/s00265-025-03649-6).
- Data and code: [zenodo.org/records/16961361](https://zenodo.org/records/16961361), a zip of release v1.0.0 of [github.com/tamarlok/postfledging_care](https://github.com/tamarlok/postfledging_care). The data files are in [data/raw](https://github.com/tamarlok/postfledging_care/tree/main/data/raw). The zip and GitHub `main` (commit 6c40d9b) hold the same data files, byte for byte.

In `/home/fatemeh/Downloads/bird/data/spoonbill_postfledging`, apart from the first paper's data: only `tamarlok-postfledging_care-6c40d9b/data/raw/` of the Zenodo zip is kept; the R code is on GitHub. The data files:

| file                                                                                                                       | what                                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `annotated.begging.data.csv`                                                                                               | the only labeled IMU data: juvenile 6381, 2018-07-04 and 05, 58 fixes, 9040 records. Columns as above, but text labels, an `annotation.method` column, and `Index` from 1 |
| `gps.behav.data.csv`                                                                                                       | 29 birds, a GPS fix every 10 min, with `behaviour` predicted by the paper's random forest, and ODBA (overall dynamic body acceleration). No x, y, z                       |
| `gps.data.csv`                                                                                                             | the same birds, GPS only                                                                                                                                                  |
| `metadata.tagged.birds.csv`, `parent.offspring.data.csv`, `chick.biometrics.csv`, `feeding.observations.with.sex.info.csv` | tags, families, chick sizes, field observations                                                                                                                           |

The raw IMU of the other birds is on Movebank, on request only ([README](https://github.com/tamarlok/postfledging_care/blob/main/README.md)).

Segments, as above. From the `# Begging data` output of `exps/spoonbill_data_stats.py`; `video` and `graphical` split the segments by `annotation.method`.

| name           | records  | bouts  | bouts < 20 | segments | % records used | video   | graphical |
| -------------- | -------- | ------ | ---------- | -------- | -------------- | ------- | --------- |
| beg            | 8861     | 62     | 4          | 429      | 97             | 100     | 329       |
| beg_flap       | 84       | 3      | 1          | 3        | 71             | 3       | 0         |
| preen_tag      | 81       | 1      | 0          | 4        | 99             | 4       | 0         |
| shake_feathers | 14       | 1      | 1          | 0        | 0              | 0       | 0         |
| **total**      | **9040** | **67** | **6**      | **436**  | **96**         | **107** | **329**   |

- The table is the labels as released, before the second paper's map, and counts incomplete segments too: 436 segments, 434 complete (the script's `complete` column). The training file has 438 begging segments (Training file, below).
- 58 fixes give 67 bouts: 5 fixes change label inside. One fix is beg, shake_feathers, beg: 3 bouts.
- 436 segments, not 9040 / 20 = 452: a segment stays inside one bout. 320 records are lost, 82 in the 6 short bouts and 238 at the ends of longer bouts.
- `graphical`: labeled from the signal alone, with the parent seen nearby in the field. Only begging was labeled this way. The R script says the signals are so clear.
- 20 Hz: the R script divides record counts by 20 to get seconds.
- 10 records lack z, 5 of them also y. Speed is never missing; at most 0.92 m/s.

The paper already combines both datasets, in [`2_analyses/1_classification_begging_behaviour.R`](https://github.com/tamarlok/postfledging_care/blob/main/2_analyses/1_classification_begging_behaviour.R):

- Map: shake_feathers to stand-shake-feathers, preen_tag to stand-preen, beg_flap to beg.
- Pool: sit-alert to sit; stand-alert, stand-other, stand-rest, stand-shake-feathers, stand-preen and drink to stand. It also scores five groups: forage, walk, fly, rest, beg.
- Segments of 0.4, 0.8, 1.0 and 1.6 s; the final random forest uses 1.6 s.

#### How the papers score

Both papers use the same method, from the code of the first ([zenodo.org/records/7541131](https://zenodo.org/records/7541131), `functions.R`):

- Random forest on 31 summary statistics per segment (mean, min, max, trend, ODBA, ...) plus GPS speed. ODBA is overall dynamic body acceleration.
- Segments cut on a fixed grid from the first record of a fix, whatever the labels in them; a segment takes its most frequent label.
- Split 70/30 at random **per segment**, so one fix can be in train and test. 100 runs, mean reported.
- Scored **per record**: each record of a test segment is compared with its own label. The F-measure is F1, on 0 to 1.

| paper | classes                                                                     | where the numbers are                                                |
| ----- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 2023  | 9: stand, sit, fly-active, fly-passive, walk, search, handle, ingest, drink | text: best over segment lengths 0.2-2.0 s; per length only in Fig. 3 |
| 2025  | 5 groups: forage, walk, fly, rest, beg                                      | Table S2 of the supplement: 0.4, 0.8 and 1.6 s                       |

Their names for our labels: search is `for-search`, handle `for-handle`, ingest (prey ingestion) `for-intake`, fly-active `fly-flap`, fly-passive `fly-soar`, sit `sit-alert`. Their stand is our five `stand-*` labels. In 2025, forage is search, handle and ingest, and rest is sit, stand and drink.

The values are in `exps/eval_spoonbill.py`, beside ours. Local copies: `/home/fatemeh/Downloads/bird/papers/Lok2023_spoonbill_prey_ingestion.pdf` (from [edepot.wur.nl/587821](https://edepot.wur.nl/587821)) and `Lok2025_spoonbill_postfledging_supplement.pdf` (Table S2).

#### Training file

`scripts/data/prepare_spoonbill_data.py` writes both datasets as one file in the `starts.csv` format: `/home/fatemeh/Downloads/bird/data/spoonbill_final/spoonbill_starts.csv`, 5080 segments, 101,600 records.

How the segment counts above add up to 5080, all in segments of 20 records:

| source                         | segments, tables above | complete, tables above | in the training file |
| ------------------------------ | ---------------------- | ---------------------- | -------------------- |
| first paper: video and soaring | 4718                   | 4642                   | 4642                 |
| begging                        | 436                    | 434                    | 438                  |
| **total**                      | **5154**               | **5076**               | **5080**             |

- Only complete segments go in: no missing x, y, z or speed.
- Begging has 4 more than its 434: the labels are mapped before the segments are cut, so a beg_flap bout joins the beg bout next to it and leftover records make new segments.
- Training splits the 5080: 4566 train and 514 valid segments, split per segment (exp205); 4575 and 505, split by fix (exp206 onward). Split by fix, the 505 valid segments come from 79 fixes and the 4575 train segments from the other 702 (`split_by_fix` prints both).
- Labels: the codes of `behaviours.csv`. Begging gets the second paper's map, and beg the new code 17. 14 labels, no pooling.
- Begging `Index` is written from 0, as in the other files.
- Segments per label and bird: the script's output. Per label they equal the `complete` column of the first table, plus 434 beg and 4 stand-preen from begging.

#### Spoonbill to gull labels

A table by Julia, an ecologist, received by email on 2026-11-16:
`/home/fatemeh/Downloads/behaviours_spoonbills_to_gulls.xlsx`, one sheet. One
row per spoonbill behaviour of Lok et al., with whether gulls show it, the gull
class she expects a model to give, an alternative, the general activity
(Flight, Sit-Stand, Terrestrial locomotion, Float, Other) and comments. Gull
names are those of Table S1 of the Judy supplement: Stationary is our SitStand,
Walk TerLoco, Peck Pecking, Mixed Manouvre. Copied into
`exps/eval_cross_species.py::SPOONBILL_TO_GULL`.

- fly-flap can be Flap, ExFlap or Manouvre (three rows).
- The preening, washing, shaking and stand-other labels go to Other, which our
  gull data drops (label 7); their alternative is Stationary.
- for-search, head swung in the water, is not found in gulls; her guess is
  Peck, or Walk or Other.
- drink is Stationary, or Other or Peck by its head movements.
- Float and Boat have no spoonbill row: spoonbills seldom float or sit on boats.
- beg is not in the table.
- She hopes a model trained on both species splits gull Stationary into rest
  and alert: alike in posture, different in meaning.

### Previous data preparation (replaced)

How the labeled data was made before the current pipeline (Labeled data,
above). Kept to explain old runs and files.

#### What was wrong with it

- Combining items into bursts: We used the first label as the label of the burst. Now, priority labels are defined in `get_rules`.
- The StandForage label was treated as Pecking. Now it is removed.
- There was a bug in m_data. The labels were shifted if the items did not start from the beginning index. A maximum of 52 labels could have been wrong (`test_issue_previous_m_data_format`). Note that this is an upper bound. This is a minor issue.
- Up to three items might have the same IMU and GPS speed. Previously, it was assumed that only two items could have the same values. This is a very minor issue, since only a few items had this problem. It appears during reindexing when getting the index from the database.
- The algorithm used to find the closest value divisible by 20 had a minor rounding issue. In Python, rounded values are rounded to the closest even number. For example, 2.5 is rounded to 2.0 instead of 3.0.

#### Old steps

The previous data preparation shifted the data to the closest value divisible by 20. In the end, each data entry contained a burst of 20.

Data preparation is in `exps/prepare_labeled_data.py`. Here are the steps:

- Convert the original data to the CSV format
- Get all data from the database
- Map to the closest value divisible by 20 and get the index from the database. Data was previously downloaded using device IDs and dates.
- Combine data (`combined.csv`)
- Remove duplicates (`combined_unique.csv`)
- Sort data by device and date in increasing index order (`combined_unique_sorted012.csv`).
- Manually correct labels by experts. The ground-truth IMU figures were generated by `exps/save_plots_gt.py`. The corrections are in the `corrections.txt` file and resulted in `corrected_combined_unique_sorted012.csv`.

> Balanced Data (`balance.csv`): For some experiments, I balanced the data by removing some rare classes and making the remaining classes have an equal number of instances based on the smallest class size. It is generated in `data.py::balance_data` by keeping {0: 'Flap', 1: 'Soar', 2: 'Float', 3: 'SitStand', 4: 'TerLoco', 5: 'Pecking'}.

In total, an extra 1235 bursts (c_data - s_data) were obtained. The "Other" class is excluded.

```
s_data: 3478  (published data, set1 JSON format)
j_data: 1456  (Suzanne JSON)
m_data: 857   (Suzanne MATLAB)
w_data: 3341  (Willem CSV)
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

#### Older methods

Kept in the `sandbox_prepare_label_data` branch, as `scripts/prepare_labeled_data.py` there.

Older methods to generate data are:

- Method1: combine only based on device ID and dates. This method is missing a lot of data.
- Method2: combine with device ID, dates, and labels. This method has a bug.

These methods are slow because they request data directly from the database. Method1 and Method2 share the database retrieval step, but the combination step differs.

The combination step in Method2 is the same as the current method in `scripts/data/prepare_labeled_data.py`.

> The bug in Method2 is in `write_j_data` and `write_m_data`: the data might contain indices 0-19 and 40-59 with label1, and 20-39 with label2. Then 0-40 gets label1, and 20-40 gets label2. This happens because the maximum length is calculated based on common labels.

## Models

Parameter counts, from the code below, at 9 classes unless noted. The working
model and how it is trained: `docs/methods.md`, Model and Training.

| model                   | arguments          | parameters                  | receptive field, time steps |
| ----------------------- | ------------------ | --------------------------- | --------------------------- |
| `BirdModelSmallDilated` | 4, 20, 9           | 5,129                       | 25                          |
| `BirdModelSmallDilated` | 7, 20, 9           | 5,429                       | 25                          |
| `BirdModelWideRF`       | 4, 20, 9           | 6,889                       | 19                          |
| `BirdModel`             | 4, 30, 9           | 6,309 (6,340 at 10 classes) | 7                           |
| `ResNet18_1D`           | 9, dropout=0.3     | 61,289                      |                             |
| `MaskedAutoencoderViT`  | img_size=20, below | 9,557,508                   |                             |
| `TransformerEncoderMAE` | below              | 4,748,297                   |                             |

```python
from behavior import model as bm, model1d as bm1
model = bm.BirdModelSmallDilated(4, 20, 9)  # or (7, 20, 9)
model = bm.BirdModelWideRF(4, 20, 9)
model = bm.BirdModel(4, 30, 9)
model = bm.ResNet18_1D(9, dropout=0.3)
model = bm1.MaskedAutoencoderViT(img_size=20, in_chans=4, patch_size=1, embed_dim=256, depth=6, num_heads=8, decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8, mlp_ratio=4, layer_norm_eps=1e-6)
model = bm1.TransformerEncoderMAE(img_size=20, in_chans=4, out_chans=9, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4, drop=0.0, layer_norm_eps=1e-6)
f"{sum([p.numel() for p in model.parameters()]):,}"
```

`BirdModel` (`behavior/model.py`) was the first model: three convolutions of
kernel 3 and 30 channels, each with BatchNorm and ReLU, dropout 0.25, average
pooling over time and one linear layer. `BirdModelSmallDilated` replaced it from
exp194 (`lesson_learned.md`, the receptive-field lesson).

## List of scripts

#### App

- `app/gps_burst_labeling_viz_app.py`: the IMU and behaviour viewer and the GPS
  map in one app, with label editing.
- `app/bird_behavior_viz_app.py`: IMU data with behaviour labels, and the
  locations on a map.
- `app/gps_locations_viz.py`: GPS traces on an interactive map.
- `app/testapp.py`, `app/testgradio.py`: early tests of a web service (FastAPI)
  and a Gradio front end that classify a CSV file with `BirdModel`. No
  docstring; read from the code.

The apps read the files of `scripts/data/bird_behavior_app_data.py` and
`scripts/data/merge_simon_data.py`, in this format:

```bash
device_id, date_time,index,groun_truth label,imu_x,imu_y,imu_z,gps_m/s,label,confidence,latitude,longitude,altitude(optional)
298,2010-06-07 09:43:05,0,-1,-0.167287,-0.004737,1.014680,0.200996,5,0.986863,53.009282,4.717725
```

#### Visualization

- `scripts/embeddings_plot.py`: t-SNE of the learned features, for a dataset
  and a model.
- `exps/save_plots_gt.py`: IMU plots of the labeled bursts, by device and start
  time.
- `exps/save_plots.py`: IMU plots per label, with the model's predictions.
- `exps/plot_errors.py`: one PNG per valid burst a model gets wrong, in
  `<true class>/<predicted class>,<device>,<datetime>,<start index>.png`, from
  the split and predictions of `exps/eval_labeled.py`. Prints where each class
  went.
- `behavior/utils.py::generate_per_glen_figures_for_dt`: IMU plots of data cut
  into bursts of 20 records, such as `shift.csv`.
- `exps/imu_3d_movie.py`: a movie of IMU data.
- `exps/figures_for_paper.py`: image grids for a paper. No docstring; read from
  the code.
- `exps/visualize.py`: old (I should remove)

#### Training

- `scripts/batch_train_supervised.py`: the training script. Several runs in one
  go, one per entry of `experiments`, each overriding `base_config`. Writes
  `<exp>_best.pth` (the epoch with the best valid accuracy), `app_loss_acc.txt`,
  and the confusion matrices and per-class metrics of train and valid in
  `save_path/failed/<exp>_<data stem>/`. The `Combined` row of
  `per_class_metrics_*.csv` equals accuracy (`status.md`, How the numbers are
  measured). Refuses an experiment number that already has results, unless
  `overwrite`. Config switches:
  - `add_magnitudes`: 7 input channels; must match
    `model.parameters.in_channels`.
  - `augment`: False turns the augmentation off. Which one is the `transforms`
    object in `main()`; it sits on the train loader only.
  - `split_file`: borrow another data file's valid split, so two runs on two
    data files are scored on the same bursts.
  - `split_seed`: the seed of the split alone (default `seed`), so the split
    holds while `seed` moves the training.
  - `split_by`: `fix` keeps every fix whole, in train or in valid
    (`split_by_fix`, scikit-learn `StratifiedGroupKFold`).
    `exps/eval_labeled.py` does not know it yet.
  - `save_final`: also keep the last epoch, `<exp>_<epoch>.pth`.
  - `label_names`: class names for other label codes, as for the spoonbills;
    None means the gull names.
- `scripts/train_joint.py`: one model on gulls and spoonbills together
  (exp212-213). 18 joint classes, the finer of the two species where they
  differ. Each species label is the set of joint classes it can be
  (`GULL_TO_JOINT`, `SPOONBILL_TO_JOINT`, from Spoonbill to gull labels,
  above), and the loss is minus the log of the probability summed over that
  set. Each species keeps the split of its own models, checked by reproducing
  the valid accuracy of exp197 and exp207. Writes the checkpoints, tensorboard
  and, in `failed/<exp>_joint/`, accuracy per species, counts per label and
  joint class, each also drawn as a confusion matrix
  (`counts_<checkpoint>_<split>.png`, `plot_counts`: an orange box marks the
  label's set), and F1 per label. A new map needs new constant names, so that
  `exps/eval_joint.py` still reads exp212-213 right.
- `scripts/train.py`: the older supervised training script.
- `scripts/train_sup_contrastive.py`: supervised classification plus a
  supervised contrastive loss (`SupConLoss`, `contrast_mode="one"`) on the
  model's features; no augmentation by default. Holds the SimGCD-style
  `DINOHead` and `DistillLoss` helpers, unused in the default run.
- `scripts/train_self_distill.py`: self-distillation with the DINO loss
  (positive pairs only). InfoNCE is not used: with 9 classes it makes false
  negatives.
- `scripts/ss_cluster_behavior.py`: semi-supervised clustering.
  `scripts/batch_train_cluster.py` runs `scripts/train.py` and it for several
  experiments. `scripts/gcd_pipeline.py`: the GCD (generalized category
  discovery) pipeline, after SimGCD.
- `scripts/classify_birds.py`: classify the bursts of given devices and time
  ranges with `BirdModel`; older, configured with `argparse`. No docstring; read
  from the code.
- Older: `exps/cluster_behavior.py` (unsupervised clustering), `exps/exps1.py`
  (inference and metrics).

#### Pretraining and foundation models

For the pretraining TODO in `status.md`. Only `exps/simGCD.py` has a docstring;
the rest is read from the code.

- `scripts/main_pretrain.py`: pretrain `MaskedAutoencoderViT` (MAE).
  `scripts/main_finetune.py`: fine-tune `TransformerEncoderMAE`.
  `scripts/pretrain_memory_load.py` and `scripts/finetune_memory_load.py` do the
  same, reading the CSV files into memory.
- `scripts/moment_pretrain.py`, `scripts/moment_finetune.py`: pretrain and
  fine-tune MOMENT.
- `exps/moment_test.py`: MOMENT features of the bursts, and a classifier on
  them. `exps/chronos2.py`: Chronos-2 features of the bursts, plotted. Neither
  has a recorded result.
- `exps/simGCD.py`: self-supervised training without negative pairs (DINO and
  similar).
- `exps/distribution_metrics.py`: overlap measures between two point clouds
  (Gaussian, convex hull, raster, kernel density).

#### Data augmentation

`behavior/data_augmentation.py`. Each transform has a per-sample form, on one
`(T, C)` burst (T=20, C=4: IMU x, y, z and GPS speed) in
`BirdDataset.__getitem__`, and a batched form, on a channel-first `(N, C, T)`
batch on the GPU, used by `bd.GpuBatches` and
`scripts/batch_train_supervised.py`. Most leave the GPS channel (column 3)
alone.

- `RandomJitter`, `RandomScaling` (and `Batch*`): Gaussian noise per element,
  scaling per channel.
- `TimeWarp`, `MagnitudeWarp` (and `Batch*`): warping in time and in magnitude.
- `RandomRotation3D`, `BatchRandomRotation3D`: a random 3D rotation of the three
  acceleration channels (`random_rotation_matrix`, a normalized Gaussian
  quaternion, uniform over rotations). The GPS speed, and the length of the
  acceleration at each time step, are unchanged.

What each costs or buys, and why augmentation must stay on the train data:
`lesson_learned.md`, Model and training and Engineering pitfalls.

#### Data scripts

- `scripts/data/prepare_labeled_data.py`: runs the pipeline of
  `behavior/data_processing.py` (Labeled data, above).
- `scripts/data/get_data_gull_20.py`, `scripts/data/get_data_gull_cp_60.py`:
  download the unlabeled data (Unlabeled data, above).
- `scripts/data/prepare_spoonbill_data.py`: writes the spoonbill training file
  from both spoonbill datasets, in the `starts.csv` format. Cuts segments with
  `exps/spoonbill_data_stats.py::cut_segments` and keeps only complete ones.
  Prints segments per label and bird, and reads the file back with
  `load_csv_pandas`. Refuses to overwrite an existing file. `load_sources`
  reads the three labeled files with labels as codes; `exps/eval_spoonbill.py`
  reuses it. See Spoonbill data, Training file.
- `scripts/data/bird_behavior_app_data.py`: app data (IMU, GPS, behaviour
  classes by inference). Set `exp` and `model_parameters.in_channels` together
  (7 for exp197, 4 otherwise); one output file per model,
  `<input stem>_<exp>.csv`. For Ornitela files, `prepare_calibrated_app_data`
  gives each burst the nearest GPS fix within 2 s (`docs/methods.md`, Ornitela
  data).
- `scripts/data/merge_simon_data.py`: app data from Simon's calibrated IMU data
  of Rose, with behaviour classes by inference. GPS is matched and x and y are
  swapped as in `prepare_calibrated_app_data`.
- `behavior/data_processing.py::create_five_balanced_data_and_unbalanced`: five
  random balanced and unbalanced datasets from one data file.
- `exps/birdvis_query.py`: a query file to import directly into the birdvis
  tool.
- `exps/check_imu_calibration.py`: checks `behavior/data.py::acc_calibrate`
  against the R output. Reads the raw `acc_x, acc_y, acc_z` of
  `all_devices_calibrated.csv` and the R parameters of
  `calibration_per_device_auto.csv`, both in `~/Downloads/bird/data/simon`, and
  prints per device the Python and R scales, and the scale with `q_static` 0.10
  against 0.85.

#### Analysis

- `exps/eval_labeled.py`: compares models on the labeled valid split: accuracy,
  average precision and loss on train and valid, pooled and per-class F1 (plain
  and class-balanced), and valid accuracy with the tag turned. Recomputes from
  the checkpoints with the functions training uses, so the numbers match
  `app_loss_acc.txt` and the training CSVs. Config: `model_exps`, `in_channels`
  (7 for the magnitude models), `checkpoints` (for example `<exp>_4000.pth`),
  and `split_file`, as in training. Prints markdown tables for `status.md`, with
  the valid bursts per class.
- `exps/eval_unlabeled.py`: compares models on unlabeled data in the app format,
  with four label-free checks: speed conflict, place conflict, unseen flip and
  rotation flip (defined in `status.md`). Runs each model in `model_exps`
  itself; `in_channels` and `checkpoints` as above. Prints a table and the
  predicted class distribution, and writes a CSV of the lowest-confidence
  bursts that fail a check, to open in the app.
- `exps/label_cooccurrence.py`: which behaviours follow one another inside a
  fix, from `combined.csv`: the unordered co-occurrence and the adjacent
  transitions, symmetrised. Writes `label_transitions.csv`, which
  `exps/eval_unlabeled.py` reads.
- `exps/find_label_noise.py`: flags bursts whose GPS speed contradicts their
  label's definition in Table S1 of the Judy supplement
  (`/home/fatemeh/Downloads/bird/papers/Judy_features_supp1.pdf`). Prints each
  rule's threshold beside the speed distribution of its class, a model's
  disagreement beside its base rate, and the flags per device and day. Writes
  `label_noise_candidates.csv`, keyed by `device_id, datetime, start_index`.
- `exps/eval_spoonbill.py`: compares the spoonbill runs with the two papers
  (How the papers score, above). Pools the 14 labels into each paper's classes
  and prints F1 (0-1) per class, mean and range over the seeds of each arm in
  `arms`, beside the paper values. Scores on records, as the papers do (1 s
  windows from the first record of each valid fix), and on the valid segments.
  Reads `<exp>_4000.pth`, after checking that each best checkpoint reproduces
  its `app_loss_acc.txt`.
- `exps/eval_cross_species.py`: each species' model on every segment of the
  other species: exp197 on `spoonbill_starts.csv`, the spoonbill runs on
  `starts.csv`. Scores each class by the label map (Spoonbill to gull labels,
  above): % given the expected class, the expected or alternative class, and
  the same general activity; the spoonbill models with the map read backwards.
  Prints a count table of where each class goes, the mean x, y and z per class
  of both species, and each model's accuracy on its own species as a check.
- `exps/eval_joint.py`: the joint models (exp212-213) beside the
  single-species models, on the same valid segments, at epoch 4000: accuracy
  and F1 per label for each species, and where the joint models put gull
  SitStand, gull Pecking and spoonbill fly-flap among their joint classes.
- `exps/spoonbill_data_stats.py`: counts the spoonbill data: records, fixes and
  data quality per file, the 20 Hz check, and segments per behaviour and per
  bird, with the bouts too short for one segment. Prints only; tables in
  Spoonbill data, above.
- `exps/speed_reliance.py`: how much a model relies on GPS speed. Keeps each
  burst's acceleration and changes only its speed: one burst swept from 0 to 20
  m/s, one class at 0 and at 20 m/s, all speeds shuffled between bursts, and all
  speeds set to 0. Builds the input through `bd.BirdDataset`, as training does,
  and loads models with `exps/eval_labeled.py`.
- `exps/magnitude_features_prior.py`: a random forest on burst summaries in
  place of the network, to test the magnitude channels without training.
- `exps/recompute_doc_numbers.py`: recomputes numbers the docs quote that no
  other script prints, one check each: the split shift when the data file
  changes, GPS speed per class, the ExFlap valid bursts that flip to Flap, the
  Ornitela GPS fixes before and after the nearest-fix rule, and the bursts and
  rows the app files drop and clip, and exp194's confidence when the tag is
  turned. `checks` picks which to run. Prints only; the results are in its
  docstring.
- `exps/inspect_unlabeled_data.py`: builds and inspects the 20-length data.
  `curate_data` applies rules 7-9 of `docs/methods.md` and the Parquet files are
  written per device; `plot_hist_scatter_plots` and `write_stats` write the
  per-device histograms, scatter plots and min and max to
  `/home/fatemeh/Downloads/bird/data/ssl/hist_ssl20`. Also holds commented-out
  one-offs. Run by editing the call blocks.
- `exps/panda_csv_python_read_data_in_memory.py`: pandas, the `csv` module and
  pure Python reading large CSV files into memory (Reading large CSV files,
  above).

#### Checks in the tests and the code

- `tests/test_data_processing.py::test_labels_comes_together`: which labels
  appear together, from the index ranges of each label per device and time
  (`get_label_ranges`, `write_all_start_end_inds`).
- `behavior/data_processing.py::identify_mistakes`: the same data labeled
  differently in two datasets. `correct_mistakes`, part of the pipeline, fixes
  them.
- `tests/test_data.py::test_find_index_jumps`: gaps in the indices: labeled
  stretches of a signal with unlabeled ones between them.
- `tests/test_data_processing.py::test_issue_previous_m_data_format`: how much
  the old m_data bug matters; not much.
- `behavior/data_processing.py::check_batches`: each burst of 20 records has
  consecutive indices, one value in columns 0, 1, 3 and 7, and no duplicate.

## List of Functions

- `behavior/data.py::GpuBatches`: keeps a whole `BirdDataset` on the GPU and
  yields shuffled batches, applying a batched transform per batch. Used in place
  of a `DataLoader`: `train_one_epoch` only iterates its loader.
- `behavior/data.py::add_magnitude_features`: `N x L x 4 -> N x L x 7`,
  appending `mag` (`||a||`), `dyn_mag` (`||a - mean_t a||`) and `jerk_mag`
  (`||a_t - a_{t-1}||`, edge-replicated at `t=0`).
- `behavior/data.py::burst_keys`: one `(device id, datetime text, start index)`
  per burst, in the load order of `load_csv_pandas`. Names a burst so a split
  can be carried from one data file to another. The timestamp of
  `load_csv_pandas` cannot do this: under pandas 3 it is 1000 times too small.
- `behavior/data.py::create_balanced_data`: a balanced dataset, an equal number
  of bursts sampled from each class.
- `behavior/data.py::acc_calibrate`: estimates the IMU calibration of one device
  from its raw data, `acc_g = (raw - b) / s` per axis. Translated from Eldar
  Rakhimberdiev's R code, which came by email with three options, worst to best:
  divide by 1058.8 (the mean of 5 Ornitela tags), one scale per device, an
  offset and a scale per axis. The R files and the email's plot are in
  `~/Downloads/bird/screenshots` (`calibration_functions 1.R`,
  `calibration_example 1.R`, `imu_calibration_plot.png`). The R functions
  become `detect_static`, `orientation_coverage`, `fit_scale_only`,
  `fit_diag_offset_bounded` and `acc_apply`; the multi-device mode and
  `acc_apply_calibration`, a copy of `acc_apply`, are left out. `get_data` calls
  it through `estimate_calibration` when the database has no calibration values.
  Rules in `docs/methods.md`, rule 1.
- `behavior/utils.py::stratified_split`: the class-wise data split.
- `behavior/utils.py::equal_dataframe`: compares two data frames.
- `behavior/map.py`: a map from latitude and longitude.
