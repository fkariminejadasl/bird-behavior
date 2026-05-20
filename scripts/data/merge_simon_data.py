#!/usr/bin/env python3
"""Merge Simon IMU, prediction, and GPS data into one headerless CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_DIR = Path("/home/fatemeh/Downloads/bird/data/simon")
DEFAULT_BIRDS_FULL = DEFAULT_DIR / "birds_full.txt"
DEFAULT_RESULTS = DEFAULT_DIR / "results.csv"
DEFAULT_CALIBRATED = DEFAULT_DIR / "all_devices_calibrated.csv"
DEFAULT_OUTPUT = DEFAULT_DIR / "simon_merged.csv"

BIRDS_FULL_COLUMNS = 8
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


def clean(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    return "" if value in {"None", "NA"} else value


def load_predictions(results_file: Path) -> dict[tuple[str, str], tuple[str, str]]:
    predictions = {}

    with results_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (clean(row["device_info_serial"]), clean(row["date_time"]))
            prediction = NAME2IND[clean(row["prediction"])]
            confidence = clean(row["confidence"])
            predictions[key] = (prediction, confidence)

    return predictions


def load_shifted_gps(
    calibrated_file: Path,
) -> dict[tuple[str, str], tuple[str, str, str]]:
    gps = {}
    gps_from_previous_row: tuple[str, str, str, str] | None = None

    with calibrated_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            device_id = clean(row["device_id"])
            utc_datetime = clean(row["UTC_datetime"])

            # GPS timestamps are one row earlier than the matching IMU timestamp.
            # Store GPS coordinates when we see a GPS row, then attach them to the
            # next row's UTC_datetime.
            if gps_from_previous_row is not None:
                gps_device_id, latitude, longitude, altitude = gps_from_previous_row
                if device_id == gps_device_id:
                    gps[(gps_device_id, utc_datetime)] = (latitude, longitude, altitude)
                gps_from_previous_row = None

            if clean(row["datatype"]) == "GPS":
                latitude = clean(row["Latitude"])
                longitude = clean(row["Longitude"])
                altitude = clean(row["Altitude_m"])
                if latitude and longitude:
                    gps_from_previous_row = (device_id, latitude, longitude, altitude)

    return gps


def merge_simon_data(
    birds_full: Path,
    results: Path,
    calibrated: Path,
    output: Path,
) -> None:
    predictions = load_predictions(results)
    gps = load_shifted_gps(calibrated)
    counts: Counter[str] = Counter()

    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        birds_full.open("r", newline="") as birds_file,
        output.open("w", newline="") as output_file,
    ):
        reader = csv.reader(birds_file)
        writer = csv.writer(output_file, lineterminator="\n")

        for row in reader:
            if len(row) < BIRDS_FULL_COLUMNS:
                counts["short_bird_rows"] += 1
                continue

            bird_row = [clean(value) for value in row[:BIRDS_FULL_COLUMNS]]
            key = (bird_row[0], bird_row[1])
            prediction, confidence = predictions.get(key, ("", ""))
            latitude, longitude, altitude = gps.get(key, ("", "", ""))

            if not prediction or not confidence:
                counts["missing_prediction"] += 1
            if not latitude or not longitude:
                counts["missing_gps"] += 1

            writer.writerow(
                [*bird_row, prediction, confidence, latitude, longitude, altitude]
            )
            counts["written"] += 1

    print(f"Wrote {counts['written']:,} rows to {output}")
    if counts["missing_prediction"]:
        print(f"Missing predictions: {counts['missing_prediction']:,}")
    if counts["missing_gps"]:
        print(f"Missing GPS rows: {counts['missing_gps']:,}")
    if counts["short_bird_rows"]:
        print(f"Skipped short bird rows: {counts['short_bird_rows']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--birds-full", type=Path, default=DEFAULT_BIRDS_FULL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--calibrated", type=Path, default=DEFAULT_CALIBRATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    merge_simon_data(args.birds_full, args.results, args.calibrated, args.output)


if __name__ == "__main__":
    main()
