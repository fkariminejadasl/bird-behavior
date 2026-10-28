# Database

Snippets for the e-ecology (UvA-BiTS) database. The first three queries are
the ones `behavior/data.py` runs (`fetch_calibration_data`, `fetch_gps_data`,
`fetch_imu_data`). The code treats database times as UTC.

## Times

```python
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

## Queries

```python
# database_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
database_url = f"postgresql://{username}:{password}@pub.e-ecology.nl:5432/eecology"
```

```python
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
