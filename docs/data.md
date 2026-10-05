# Example Times

```bash
>>> import pytz # pip install pytz
>>> from datetime import datetime, timezone

>>> dt_format = datetime.strptime('2023-11-06 14:08:11', "%Y-%m-%d %H:%M:%S")
datetime.datetime(2023, 11, 6, 14, 8, 11)
>>> dt_format.strftime("%Y-%m-%d %H:%M:%S")
>>> datetime(2023, 11, 6, 14, 8, 11).strftime("%Y-%m-%d %H:%M:%S")
'2023-11-06 14:08:11'

>>> datetime.strptime('2023-11-06 14:08:11.915636', "%Y-%m-%d %H:%M:%S.%f").timestamp()
>>> datetime.strptime('2023-11-06 13:08:11.915636', "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
1699276091.915636
>>> datetime.fromtimestamp(1699276091.915636, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
'2023-11-06 13:08:11.915636'

# Specify timezone when running in cloud
>>> datetime.fromtimestamp(1416956654.0, pytz.timezone('CET')).strftime("%Y-%m-%d %H:%M:%S")
'2014-11-26 00:04:14'
>>> datetime.fromtimestamp(1416956654.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
'2014-11-25 23:04:14'
>>> pytz.timezone('CET')
<DstTzInfo 'CET' CET+1:00:00 STD>
```

# Example Database Queries

```database format:
# database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
database_url = f"postgresql://{username}:{password}@pub.e-ecology.nl:5432/eecology"
```

```bash
device_id = 805
start_time = '2015-05-27 09:19:34' 
end_time = '2015-05-27 09:20:34'

# Get calibration imu values from database
cal_query = f"""
select *
from gps.ee_tracker_limited
where device_info_serial = {device_id}
"""

# speed_2d for gpd speed
gps_query = f"""
SELECT *
FROM gps.ee_tracking_speed_limited
WHERE device_info_serial = {device_id} and date_time between '{start_time}' and '{end_time}'
order by date_time
"""

# get imu
imu_query = f"""
SELECT *
FROM gps.ee_acceleration_limited
WHERE device_info_serial = {device_id} and date_time between '{start_time}' and '{end_time}'
order by date_time, index
"""

device_query = """
select device_info_serial 
from gps.ee_tracker_limited
"""

device_start_end_query = """
select device_info_serial, start_date, end_date 
from gps.ee_track_session_limited etsl
"""
```

# Get Some Statistics

Just a short check to see if the whole data, training and validation set are balanced. The code snippet below generates these values.

```bash
((4394, 20, 4), (4365, 20, 4)) # gimus, gimus2
((3928, 3), (437, 3)) # tldts2, vldts2
[(7, 29), (1, 41), (8, 150), (3, 176), (6, 342), (9, 342), (2, 541), (4, 623), (0, 643), (5, 1507)] # ldts
[(1, 41), (7, 150), (3, 176), (6, 342), (8, 342), (2, 541), (4, 623), (0, 643), (5, 1507)] # ldts2
[(1, 37), (7, 133), (3, 150), (8, 304), (6, 316), (2, 496), (4, 560), (0, 585), (5, 1347)] # tld2
[(1, 4), (7, 17), (3, 26), (6, 26), (8, 38), (2, 45), (0, 58), (4, 63), (5, 160)] # vl2
[(1, 0.01), (7, 0.03), (3, 0.04), (8, 0.08), (6, 0.08), (2, 0.13), (4, 0.14), (0, 0.15), (5, 0.34)] # tper_abs
[(1, 0.01), (7, 0.04), (3, 0.06), (6, 0.06), (8, 0.09), (2, 0.1), (0, 0.13), (4, 0.14), (5, 0.37)]  # vper_abs
[(1, 0.11), (7, 0.13), (3, 0.17), (8, 0.12), (6, 0.08), (2, 0.09), (4, 0.11), (0, 0.1), (5, 0.12)]  # per_rel
```

```python
from behavior import data as bd
from collections import Counter
import numpy as np
target_labels = [0, 1, 2, 3, 4, 5, 6, 8, 9]
train_per, data_per = 0.9, 1.0
gimus, ldts = bd.load_csv("/home/fatemeh/Downloads/bird/data/combined_s_w_m_j.csv")
gimus2, ldts2 = bd.get_specific_labesl(gimus, ldts, target_labels)
# sorted(dict(Counter(ldts[:,0])).items(), key=lambda x:x[1])
# sorted(dict(Counter(ldts2[:,0])).items(), key=lambda x:x[1])
n_trainings = int(gimus2.shape[0] * train_per * data_per)
n_valid = gimus2.shape[0] - n_trainings
tldts2 = ldts2[:n_trainings]
vldts2 = ldts2[n_trainings : n_trainings + n_valid]
tl2 = dict(sorted(dict(Counter(tldts2[:,0])).items(), key=lambda x:x[1]))
vl2 = dict(sorted(dict(Counter(vldts2[:,0])).items(), key=lambda x:x[1]))
vper_abs = {k:round(v/437,2) for k, v in vl2.items()}
tper_abs = {k:round(v/3928,2) for k, v in tl2.items()}
per_rel = dict()
for tkey, tval in tl2.items():
	for vkey, vval in vl2.items():
		if tkey == vkey:
			per_rel[tkey] = round(vval/tval,2)
```

#### Remove other label

```python
# {0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking'}
import pandaas as pd
df = pd.read_csv(Path("/home/fatemeh/Downloads/bird/data/combined_s_w_m_j.csv"), header=None)
filtered_df = df[df[3] != 7]
filtered_df.loc[:, 3] = filtered_df[3].apply(lambda x: x if x < 7 else x-1)
filtered_df.to_csv('/home/fatemeh/Downloads/bird/data/combined_s_w_m_j_no_others.csv', header=False, index=False)
```

## Description of Data and Model

How rows from the database become the bursts the model sees, and how the model
is trained.

Words used below:

- **fix**: one GPS record of one device, at one time in whole seconds.
- **burst**: the accelerometer (IMU) samples recorded at a fix. Each sample has
  the fix time and an `index`, its place in the burst. The labeled data is
  sampled at 20 Hz.
- **model input**: 20 consecutive samples (1 s) of IMU x, y, z, plus the GPS 2D
  speed of the fix, copied to all 20.

### Data

Rules 1-6 run at download, in `behavior/data.py::get_data`. Rules 7-9 run when
a CSV is loaded.

The Gulliver repository (`~/dev/gulliver-behavior-classifier/`) gets its data
already downloaded and calibrated, with a timestamp per IMU sample instead of an
index. *Gulliver* says what a rule becomes there.

At download:

1. **Calibration is known.** The device needs calibration values in
   `gps.ee_tracker_limited`, or nothing is downloaded for it
   (`fetch_calibration_data`). Each axis goes from raw counts to g:
   `(raw - offset) / sensitivity`, rounded to 8 decimals (`raw2meas`,
   `calibrate_imu_data`).
   - *Gulliver*: already done.
2. **No missing values.** A GPS fix without `speed_2d` is dropped
   (`fetch_gps_data`). An IMU sample with x, y or z missing is dropped
   (`fetch_imu_data`).
   - *Gulliver*: drop the same rows.
3. **Indices start at 0.** If the first sample returned has index 1, every
   index is lowered by 1 (`calibrate_imu_data`). Only that first sample is
   checked.
   - *Gulliver*: not needed.
4. **Samples are consecutive.** A run goes on while each index is the one
   before plus 1. A missing sample ends it: index 4 followed by 6 splits the
   burst in two (`identify_and_process_groups`).
   - *Gulliver*: the next sample is one sample period later (50 ms at 20 Hz),
     within the same fix.
5. **Cut into bursts of 20.** A run shorter than 20 is dropped. A longer run is
   cut into 20-sample bursts from its first sample, and the rest is dropped.
   Example: a burst of 50 samples (0-49) missing index 4 keeps 5-24 and
   25-44, and drops 0-3 and 45-49 (`identify_and_process_groups`).
   - *Gulliver*: the same.
6. **GPS and IMU match.** A burst is kept only if a GPS fix has the same time,
   to the second. Its speed (and latitude, longitude, altitude, temperature) is
   copied to every sample of the burst. If two fixes match, the first is used
   (`match_gps_to_groups`).
   - *Gulliver*: match on the fix time. After rule 5 a burst can start after
     sample 0, so its first timestamp can be later than the fix.

At load time (`behavior/data.py::load_csv_pandas` for training; the same lines
are in `exps/inspect_unlabeled_data.py::curate_data` and
`scripts/data/bird_behavior_app_data.py`):

7. **GPS speed below 30 m/s.** Faster rows are dropped. All samples of a burst
   share one speed, so whole bursts go. Above 30 m/s the speed is sensor error:
   raw speeds reach 521 m/s on the unlabeled data
   (`exps/inspect_unlabeled_data.py`). No labeled burst is that fast.
8. **IMU clipped to [-2, 2] g.** This bounds outliers: raw IMU reaches 15.75 g
   on the unlabeled data, and 782 of 86,760 labeled rows are outside.
9. **GPS speed divided by 22.3012351755624**, the largest speed in the labeled
   data, so it lies in about [0, 1] (`behavior/data.py::BirdDataset`).
   - *Gulliver*: rules 7-9 the same, with this same number. Do not divide by
     the largest speed of the new data: the model learned this scale.

Also to know:

- `get_data` does not check the sampling rate. A burst at another rate passes
  every rule, but 20 samples are then not 1 s.
- Units: IMU in g, GPS speed in m/s. Ornitela gives km/h (divide by 3.6) and
  swaps x and y compared with the UvA-BiTS loggers of the labeled data; both
  are undone in `scripts/data/bird_behavior_app_data.py`.
- Labeled data: `get_data` runs once per labeled fix
  (`behavior/data_processing.py::get_s_j_w_m_data_from_database`). Each run of
  one label is then cut into bursts of 20 from its first labeled sample, as in
  rule 5 (`behavior/data_processing.py::slice_from_first_label`), giving
  `starts.csv`. The full pipeline is in `docs/descriptions.md`, Labeled data.

### Model

`behavior/model.py::BirdModelSmallDilated`. The working model is exp197.

- Input: one burst, channels x 20 samples. 4 channels (IMU x, y, z, GPS
  speed), or 7 with `add_magnitudes=True`. The 3 extra channels are lengths of
  the acceleration vector, which a rotation does not change
  (`behavior/data.py::add_magnitude_features`):
  - `mag`: the total acceleration.
  - `dyn_mag`: the acceleration minus its burst mean, so gravity and posture
    are removed. This is VeDBA (vectorial dynamic body acceleration).
  - `jerk_mag`: the change from one sample to the next.
- Three 1-D convolutions, kernel 5, 20 channels, dilation 1, 2 and 3. Each is
  followed by GroupNorm (4 groups) and GELU. The receptive field is 25 samples,
  so each output sees the whole burst.
- The 20 time steps are pooled into the mean, max and standard deviation of
  each channel (60 numbers). Then dropout 0.15 and one linear layer to the 9
  classes.
- 5,129 parameters with 4 channels, 5,429 with 7.

### Training

`scripts/batch_train_supervised.py::main`. The settings are `base_config` and
`experiments` under `if __name__ == "__main__":`.

- Data: `starts.csv`, 4,338 labeled bursts, 9 classes (Other removed). Split
  90/10 within each class (`behavior/utils.py::stratified_split`, seed 32984):
  3,900 train, 438 valid.
- Loss: cross-entropy, no class weights.
- Optimizer: AdamW, learning rate 3e-4, weight decay 1e-2. The learning rate
  is multiplied by 0.1 after epoch 2,000 (StepLR). 4,000 epochs, all train
  bursts in one batch.
- Augmentation, train set only: each burst gets a new random 3D rotation of IMU
  x, y, z every epoch; GPS is not touched
  (`behavior/data_augmentation.py::BatchRandomRotation3D`). The way a logger
  sits on a new bird is unknown, so the model should not depend on it.
- Checkpoint: the epoch with the best valid accuracy, `<exp>_best.pth`.
  `save_final=True` also keeps epoch 4,000.
- About 5 minutes on the laptop GPU (exp197: 4m56s).
- exp197: valid accuracy 94.52%, F1 0.95, class-balanced F1 0.90
  (`exps/eval_labeled.py`).
