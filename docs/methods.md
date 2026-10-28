# Methods

How raw data becomes the bursts the model sees, from the UvA-BiTS database
and from Ornitela files, and how the model is trained.

Words used below:

- **fix**: one GPS record of one device, at one time in whole seconds.
- **burst**: the accelerometer (IMU) samples recorded at a fix. In the database
  each sample has the fix time and an `index`, its place in the burst. The
  labeled data is sampled at 20 Hz.
- **model input**: 20 consecutive samples (1 s) of IMU x, y, z in g, plus the
  GPS 2D speed of the fix in m/s, copied to all 20.

## Database data (UvA-BiTS)

Rules 1-6 run at download, in `behavior/data.py::get_data`. Rules 7-9 run when
a CSV is loaded, for this data and for the Ornitela data below.

[Gulliver](https://github.com/fkariminejadasl/gulliver-behaviour-classifier)
gets this data already downloaded and calibrated, with a timestamp per IMU
sample instead of an index. *Gulliver* says what a rule becomes there.

At download:

1. **Calibration.** Each axis goes from raw counts to g:
   `(raw - offset) / sensitivity`, rounded to 8 decimals (`raw2meas`,
   `calibrate_imu_data`). The values come from `gps.ee_tracker_limited`
   (`fetch_calibration_data`). A device with no values there, or a missing or
   NaN value, is calibrated from its own IMU data in the requested time range
   (`estimate_calibration`, a translation of Eldar Rakhimberdiev's R code):
   - the static samples are found (`detect_static`, 2 s window);
   - with at least 300 of them, one scale for all axes, or an offset and a
     scale per axis if they cover enough postures (`acc_calibrate`). On a bird
     they rarely do, so it is usually one scale (see Calibrating a tag off
     the bird). 300 is the R default `min_static`, 15 s at 20 Hz. R gives no
     reason for it, and it was not tested here;
   - with fewer, as for a single burst, no offset and a scale of 1058.8, the
     mean of 5 Ornitela tags. A scale fitted on one burst would set its own
     magnitude to 1 g.
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
   copied to every sample of the burst. A fix even 1 s away is not used. If two
   fixes match, the first is used (`match_gps_to_groups`).
   - *Gulliver*: match on the fix time. After rule 5 a burst can start after
     sample 0, so its first timestamp can be later than the fix.

At load time (`behavior/data.py::load_csv_pandas` for training; the same lines
are in `exps/inspect_unlabeled_data.py::curate_data` and
`scripts/data/bird_behavior_app_data.py::infer_save_app_data`):

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
- **Ornitela x and y: decided later.** Ornitela data will come either into the
  database, already swapped to the UvA-BiTS axes, or as a file, not swapped
  (rule 4 below swaps those). Which one is a future decision. For now
  `get_data` does not swap them. The calibration does not depend on it: it
  works on the axes as stored.
- Labeled data: `get_data` runs once per labeled fix
  (`behavior/data_processing.py::get_s_j_w_m_data_from_database`). Each run of
  one label is then cut into bursts of 20 from its first labeled sample, as in
  rule 5 (`behavior/data_processing.py::slice_from_first_label`), giving
  `starts.csv`. The full pipeline is in `docs/descriptions.md`, Labeled data.

### Calibrating a tag off the bird

An offset and a scale per axis need the tag still in several orientations. A
bird rests in too few, so this is best done before deployment. This is
reasoning, not something tested here.

- **Six positions:** each axis pointing down, then up, each held still for
  about 10-30 s. The 2 s window drops the samples near each turn, and 300
  still samples are needed in total.
- **Per axis:** pointing up, it reads `b + s`; pointing down, `b - s` (or the
  reverse, by the sign convention of the device). So `s = (up - down) / 2` and
  `b = (up + down) / 2`. The other two axes read about their offset b (0 g).
- **Exact positions are not needed.** The fit only asks for a length of 1 g at
  every still moment, so clearly different orientations are enough. Six are
  the easiest to do and to check.
- **Example:** Tamar Lok's recording,
  `~/Downloads/bird/screenshots/imu_calibration_plot.png`: about 40 minutes
  still, then several turns.
- **Store the six values** in `gps.ee_tracker_limited`, in the order
  `x_o, x_s, y_o, y_s, z_o, z_s`. `get_data` then uses them and estimates
  nothing.

## Ornitela data

`scripts/data/bird_behavior_app_data.py::prepare_calibrated_app_data`. The input
is one calibrated CSV with GPS rows and IMU rows (`datatype` GPS or SENSORS).
The IMU is already in g. There is no index. Usually a GPS row is followed by
the IMU rows of one burst, 0-2 s later.

1. **A burst is the IMU rows of one device with the same time.** Only whole
   bursts of 20 are kept: 45 rows give 40 (two bursts), 15 rows give none. Rows
   are indexed from 0. Without an index, a missing sample cannot be found.
2. **Each burst takes the GPS fix nearest in time**, at most 2 s before or
   after. It gets the fix's speed, latitude, longitude and altitude. Only fixes
   of the same device count, and not fixes at latitude 0 and longitude 0. A
   burst with no fix within 2 s is dropped.
3. **Example.** With fixes at -2, -1, 0, +1 and +2 s, the burst takes the fix
   at 0 s. With fixes only at -1 and +1 s, it takes the earlier one, at -1 s.
4. **x and y are swapped**, to match the UvA-BiTS axes of the labeled data: the
   model's x is Ornitela `y_g`, and its y is `x_g`.
5. **Speed from km/h to m/s**: divide by 3.6.

Then rules 7-9 apply, as for the database data (`prepare_app_class_data`).
`scripts/data/merge_simon_data.py::load_calibrated_rows` uses the same rules 1-5.
`check_consecutive_gps_sensor_time_diff`, in the same script, prints how far
apart each fix and the next SENSORS row are.

## Model

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

## Training

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
