"""Merge Simon calibrated IMU and GPS, and prediction into CSV for visualization apps.

Input files
-----------
``results.csv`` is the output from classify_birds.py and must include the header
columns ``device_info_serial``, ``date_time``, ``prediction``, and
``confidence``. ``prediction`` is a behavior name from ``ind2name`` and is
converted to its numeric class ID in the merged output.

``all_devices_calibrated.csv`` is the calibrated Simon/Rose export with a
header. The script reads ``device_id``, ``UTC_datetime``, ``datatype``,
``Latitude``, ``Longitude``, ``Altitude_m``, ``speed_km_h``, ``x_g``, ``y_g``,
and ``z_g``. GPS rows are identified with ``datatype == "GPS"`` and SENSOR/IMU
rows are identified with ``datatype == "SENSORS"``. In this file the GPS
timestamp is one row earlier than the matching SENSOR/IMU timestamp, so each
valid GPS location is attached to the next SENSOR rows for the same device.

Output file
-----------
``simon_merged.csv`` is a headerless CSV consumed by the visualization apps:

    device_id,date_time,index,gt_label,imu_x,imu_y,imu_z,gps_speed,label,confidence,latitude,longitude,altitude

``gt_label`` is ``-1`` for unlabeled data. ``label`` is the predicted numeric
behavior class. Missing ``None`` or ``NA`` values are written as empty strings.
"""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path("/home/fatemeh/Downloads/bird/data/simon")
DEFAULT_RESULTS = DEFAULT_DIR / "results.csv"
DEFAULT_CALIBRATED = DEFAULT_DIR / "all_devices_calibrated.csv"
DEFAULT_OUTPUT = DEFAULT_DIR / "simon_merged2.csv"

APP_GROUP_SIZE = 20
MAX_GPS_SENSOR_TIME_DIFF_SECONDS = 2
ind2name = {
    0: "Flap",
    1: "ExFlap",
    2: "Soar",
    3: "Boat",
    4: "Float",
    5: "SitStand",
    6: "TerLoco",
    7: "Other",
    8: "Manouvre",
    9: "Pecking",
}
NAME2IND = {name: str(ind) for ind, name in ind2name.items()}


def clean(value: Optional[str]) -> str:
    if value is None:
        return ""
    value = value.strip()
    return "" if value in {"None", "NA", "NaN", "nan"} else value


def parse_datetime(value: Optional[str]) -> datetime:
    value = clean(value)
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return datetime.fromisoformat(value)


def clean_datetime(value: Optional[str]) -> str:
    value = clean(value)
    if not value:
        return ""

    return parse_datetime(value).strftime("%Y-%m-%d %H:%M:%S")


def format_imu(value: Optional[str]) -> str:
    value = clean(value)
    if not value:
        return ""

    value = float(value)
    return f"{value:.6f}" if isfinite(value) else ""


def valid_number(value: Optional[str]) -> bool:
    value = clean(value)
    if not value:
        return False

    try:
        return isfinite(float(value))
    except ValueError:
        return False


def load_predictions(results_file: Path) -> dict[tuple[str, str], tuple[str, str]]:
    predictions = {}

    with results_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (clean(row["device_info_serial"]), clean_datetime(row["date_time"]))
            prediction = NAME2IND.get(clean(row["prediction"]), "")
            confidence = clean(row["confidence"])
            predictions[key] = (prediction, confidence)

    return predictions


def is_zero(value: str) -> bool:
    try:
        return float(value) == 0
    except ValueError:
        return False


def load_calibrated_rows(calibrated_file: Path) -> list[list[str]]:
    with calibrated_file.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    latest_gps_by_device: dict[str, Optional[dict[str, str]]] = {}
    grouped_rows: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    group_order: list[tuple[str, str]] = []

    for row_index, row in enumerate(rows):
        device_id = clean(row["device_id"])
        datatype = clean(row["datatype"])

        if datatype == "GPS":
            latitude = clean(row["Latitude"])
            longitude = clean(row["Longitude"])
            next_row = rows[row_index + 1] if row_index + 1 < len(rows) else None
            latest_gps_by_device[device_id] = None

            if next_row is not None:
                time_diff = parse_datetime(next_row["UTC_datetime"]) - parse_datetime(
                    row["UTC_datetime"]
                )
                valid_gps = (
                    clean(next_row["datatype"]) == "SENSORS"
                    and clean(next_row["device_id"]) == device_id
                    and latitude
                    and longitude
                    and not (is_zero(latitude) and is_zero(longitude))
                    and abs(time_diff.total_seconds())
                    <= MAX_GPS_SENSOR_TIME_DIFF_SECONDS
                )
                if valid_gps:
                    latest_gps_by_device[device_id] = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "altitude": clean(row["Altitude_m"]),
                        "speed_km_h": clean(row["speed_km_h"]),
                    }
            continue

        if datatype != "SENSORS":
            continue

        gps = latest_gps_by_device.get(device_id)
        if gps is None:
            continue

        date_time = clean_datetime(row["UTC_datetime"])
        key = (device_id, date_time)
        if key not in grouped_rows:
            group_order.append(key)

        speed = clean(gps["speed_km_h"])
        x_g = format_imu(row["x_g"])
        y_g = format_imu(row["y_g"])
        z_g = format_imu(row["z_g"])

        if not all(
            [
                x_g,
                y_g,
                z_g,
                valid_number(speed),
                valid_number(gps["latitude"]),
                valid_number(gps["longitude"]),
            ]
        ):
            continue

        altitude = gps["altitude"] if valid_number(gps["altitude"]) else "-1"
        grouped_rows[key].append(
            [
                device_id,
                date_time,
                "",
                "-1",
                x_g,
                y_g,
                z_g,
                f"{float(speed) / 3.6:.6f}" if speed else "",
                gps["latitude"],
                gps["longitude"],
                altitude,
            ]
        )

    calibrated_rows = []
    for key in group_order:
        rows_in_group = grouped_rows[key]
        keep_n = len(rows_in_group) - (len(rows_in_group) % APP_GROUP_SIZE)
        for sample_index, row in enumerate(rows_in_group[:keep_n]):
            row[2] = str(sample_index)
            calibrated_rows.append(row)

    return calibrated_rows


def merge_simon_data(
    results: Path,
    calibrated: Path,
    output: Path,
) -> None:
    predictions = load_predictions(results)
    calibrated_rows = load_calibrated_rows(calibrated)
    counts: Counter[str] = Counter()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        for row in calibrated_rows:
            bird_row = row[:8]
            key = (bird_row[0], bird_row[1])
            if key not in predictions:
                counts["missing_prediction"] += 1
                continue

            prediction, confidence = predictions[key]

            if not prediction or not confidence:
                counts["missing_prediction"] += 1
                continue

            writer.writerow([*bird_row, prediction, confidence, *row[8:]])
            counts["written"] += 1

    print(f"Wrote {counts['written']:,} rows to {output}")
    if counts["missing_prediction"]:
        print(f"Missing predictions: {counts['missing_prediction']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--calibrated", type=Path, default=DEFAULT_CALIBRATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    merge_simon_data(args.results, args.calibrated, args.output)


if __name__ == "__main__":
    main()
