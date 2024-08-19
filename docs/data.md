# Example Times
``` bash
>>> import pytz # pip install pytz
>>> from datetime import datetime, timezone

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