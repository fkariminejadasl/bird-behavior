from pathlib import Path

import pandas as pd
import pytest

"""
This is sort of end-to-end test for classify_birds script.
"""


@pytest.mark.local
def test_bird_classification():
    # Load the ground truth files
    gt_path = Path("/home/fatemeh/Downloads/bird/data/tests/gt")
    ground_truth_1 = pd.read_csv(
        gt_path / "541_2012-05-17 00:00:59_2012-05-17 00:00:59.csv"
    )
    ground_truth_2 = pd.read_csv(gt_path / "failures.csv")
    ground_truth_3 = pd.read_csv(
        gt_path / "805_2014-06-05 11:16:27_2014-06-05 11:17:27.csv"
    )

    # Load the generated files
    # python scripts/classify_birds.py configs/classification.yaml
    result_path = Path("/home/fatemeh/Downloads/bird/data/tests/exp2")
    generated_1 = pd.read_csv(
        result_path / "541_2012-05-17 00:00:59_2012-05-17 00:00:59.csv"
    )
    generated_2 = pd.read_csv(result_path / "failures.csv")
    generated_3 = pd.read_csv(
        result_path / "805_2014-06-05 11:16:27_2014-06-05 11:17:27.csv"
    )

    # Function to compare two dataframes
    def compare_dataframes(df1, df2, cols_to_exclude=["runtime"]):
        df1 = df1.drop(columns=cols_to_exclude)
        df2 = df2.drop(columns=cols_to_exclude)
        if df1.equals(df2):
            return True
        else:
            return False

    # Compare and print results
    if compare_dataframes(ground_truth_1, generated_1):
        print("File 1 matches the ground truth.")
    else:
        print("File 1 does not match the ground truth.")

    if compare_dataframes(ground_truth_2, generated_2, []):
        print("File 2 matches the ground truth.")
    else:
        print("File 2 does not match the ground truth.")

    if compare_dataframes(ground_truth_3, generated_3):
        print("File 3 matches the ground truth.")
    else:
        print("File 3 does not match the ground truth.")
