import json
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import behavior.data as bd
import behavior.data_processing as bdp
from behavior.data_processing import (
    build_index,
    complete_data_from_db,
    evaluate_and_modify_df,
    find_matching_index,
    get_label_range,
    map_to_nearest_divisible_20,
)


@pytest.fixture
def sample_json_data():
    return [
        {
            "labelDetail": {"description": "Flap", "labelId": 1},
            "gpsRecord": {
                "deviceId": 123,
                "timeStamp": 1609459200000,
                "measurements": [
                    {"x": 0.1, "y": 0.2, "z": 0.3, "gpsSpeed": 10.1},
                    {"x": 0.4, "y": 0.5, "z": 0.6, "gpsSpeed": 10.2},
                ],
            },
        }
    ]


def test_combine_jsons_to_one_json(tmp_path, sample_json_data):
    # Create two sample json files
    file1 = tmp_path / "file1.json"
    file2 = tmp_path / "file2.json"
    output_file = tmp_path / "combined.json"

    for f in [file1, file2]:
        with open(f, "w") as fp:
            json.dump(sample_json_data, fp)

    bd.combine_jsons_to_one_json([file1, file2], output_file)

    with open(output_file) as f:
        combined_data = json.load(f)

    assert isinstance(combined_data, list)
    assert len(combined_data) == 2 * len(sample_json_data)
    assert combined_data[0]["labelDetail"]["description"] == "Flap"


def test_read_json_data(tmp_path, sample_json_data):
    json_path = tmp_path / "input.json"
    with open(json_path, "w") as f:
        json.dump(sample_json_data, f)

    labels, label_ids, device_ids, time_stamps, all_measurements = bd.read_json_data(
        json_path
    )

    assert labels == ["Flap"]
    assert label_ids == [0]  # zero-based
    assert device_ids == [123]
    assert time_stamps == [1609459200]
    assert all_measurements.shape == (1, 2, 4)


def test_get_per_location_measurements():
    sample_meas = [
        {"x": 1, "y": 2, "z": 3, "gpsSpeed": 4},
        {"x": 5, "y": 6, "z": 7, "gpsSpeed": 8},
    ]
    result = bd.get_per_location_measurements(sample_meas)
    assert result == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_load_all_data_from_json(tmp_path, sample_json_data):
    json_path = tmp_path / "input.json"
    with open(json_path, "w") as f:
        json.dump(sample_json_data, f)

    all_measurements, ldt = bd.load_all_data_from_json(json_path)

    assert isinstance(all_measurements, np.ndarray)
    assert all_measurements.shape == (1, 2, 4)
    assert ldt.shape == (1, 3)
    print(ldt)
    assert ldt[0][0] == 0  # label_id zero-based


def test_write_j_data_orig(tmp_path, sample_json_data):
    json_file = tmp_path / "input.json"
    save_file = tmp_path / "output.csv"
    with open(json_file, "w") as f:
        json.dump(sample_json_data, f)

    new2old_labels = {i: i for i in range(10)}  # identity mapping
    bdp.write_j_data_orig(json_file, save_file, new2old_labels, ignored_labels=[])

    assert save_file.exists()
    content = save_file.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2  # 2 measurements
    assert lines[0].startswith("123,2021-01-01 00:00:00,-1,0,0.100000")


def test_write_j_data_orig_with_ignored_label(tmp_path, sample_json_data):
    json_file = tmp_path / "input.json"
    save_file = tmp_path / "output.csv"
    with open(json_file, "w") as f:
        json.dump(sample_json_data, f)

    new2old_labels = {i: i for i in range(10)}
    bdp.write_j_data_orig(json_file, save_file, new2old_labels, ignored_labels=[0])

    # The label should be ignored
    assert save_file.exists()
    content = save_file.read_text()
    assert content == ""


def test_map_to_nearest_divisible_20():
    assert map_to_nearest_divisible_20(2, 18) == [0, 20]
    assert map_to_nearest_divisible_20(23, 37) == [20, 40]
    assert map_to_nearest_divisible_20(35, 55) == [40, 60]


def test_find_matching_index():
    df = pd.read_csv(
        Path(__file__).parent.parent / "data/data_from_db.csv", header=None
    )
    keys = df[(df[0] == 533) & (df[1] == "2012-05-15 05:41:52")][[4, 5, 6, 7]].values
    query = np.array(
        [
            [0.225012, -0.433472, 1.318443, 9.072514],
            [0.281927, 1.049555, 0.661937, 9.072514],
        ]
    )
    assert find_matching_index(keys, query) == 19


def test_get_label_range():
    df = pd.read_csv(
        Path(__file__).parent.parent / "data/slice_w_data.csv", header=None
    )
    device_id, start_time = 533, "2012-05-15 05:41:52"
    slice = df[(df[0] == device_id) & (df[1] == start_time)]
    label_ranges = get_label_range(slice)
    assert label_ranges == [[2, 0, 19], [8, 19, 60]]


def test_build_index_n1():
    keys = [[1], [2], [3]]
    expected = {((1,),): [0], ((2,),): [1], ((3,),): [2]}
    assert build_index(keys, n_rows=1) == expected


def test_build_index_n2():
    keys = [[1], [2], [3]]
    expected = {((1,), (2,)): [0], ((2,), (3,)): [1]}
    assert build_index(keys, n_rows=2) == expected


def test_build_index_n3():
    keys = [[1], [2], [3]]
    expected = {((1,), (2,), (3,)): [0]}
    assert build_index(keys, n_rows=3) == expected


def test_build_index_too_few_elements():
    keys = [[1]]
    expected = {}
    assert build_index(keys, n_rows=2) == expected


def test_build_index_exact_elements():
    keys = [[1], [2]]
    expected = {((1,), (2,)): [0]}
    assert build_index(keys, n_rows=2) == expected


def test_build_index_duplicates_n1():
    keys = [[1], [1], [1]]
    expected = {((1,),): [0, 1, 2]}
    assert build_index(keys, n_rows=1) == expected


def test_build_index_duplicates_n2():
    keys = [[1], [1], [1]]
    expected = {((1,), (1,)): [0, 1]}
    assert build_index(keys, n_rows=2) == expected


def test_complete_data_from_db_with_example():
    df_csv = """6073,2016-06-07 12:34:34,9,11,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,11,5,0.586372,0.191781,1.155054,0.044410
6073,2016-06-07 12:38:39,0,10,0.357965,-0.109589,1.020608,0.004212
6073,2016-06-07 12:38:39,1,10,0.324376,0.019569,0.933268,0.004212
6073,2016-06-07 12:38:39,2,10,0.228407,-0.244618,0.868499,0.004212"""

    df_db_csv = """6073,2016-06-07 12:34:34,8,-1,0.300384,0.037182,1.015702,0.044410
6073,2016-06-07 12:34:34,9,-1,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,10,-1,0.186180,0.038160,0.921492,0.044410
6073,2016-06-07 12:34:34,11,-1,0.586372,0.191781,1.155054,0.044410
6073,2016-06-07 12:34:34,12,-1,0.093090,-0.208415,0.884200,0.044410
6073,2016-06-07 12:38:39,0,-1,0.357965,-0.109589,1.020608,0.004212
6073,2016-06-07 12:38:39,1,-1,0.324376,0.019569,0.933268,0.004212
6073,2016-06-07 12:38:39,2,-1,0.228407,-0.244618,0.868499,0.004212
6073,2016-06-07 12:38:39,3,-1,0.392514,-0.155577,1.131501,0.004212
6073,2016-06-07 12:38:39,4,-1,0.351248,0.069472,0.966634,0.004212
6210,2016-05-09 11:09:51,10,-1,0.400588,-0.271930,0.916008,0.061925"""

    expected_df_csv = """6073,2016-06-07 12:34:34,8,-1,0.300384,0.037182,1.015702,0.044410
6073,2016-06-07 12:34:34,9,11,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,10,-1,0.186180,0.038160,0.921492,0.044410
6073,2016-06-07 12:34:34,11,5,0.586372,0.191781,1.155054,0.044410
6073,2016-06-07 12:34:34,12,-1,0.093090,-0.208415,0.884200,0.044410
6073,2016-06-07 12:38:39,0,10,0.357965,-0.109589,1.020608,0.004212
6073,2016-06-07 12:38:39,1,10,0.324376,0.019569,0.933268,0.004212
6073,2016-06-07 12:38:39,2,10,0.228407,-0.244618,0.868499,0.004212
6073,2016-06-07 12:38:39,3,-1,0.392514,-0.155577,1.131501,0.004212
6073,2016-06-07 12:38:39,4,-1,0.351248,0.069472,0.966634,0.004212"""

    df = pd.read_csv(StringIO(df_csv), header=None)
    df_db = pd.read_csv(StringIO(df_db_csv), header=None)
    expected_df = pd.read_csv(StringIO(expected_df_csv), header=None)

    result_df = complete_data_from_db(df, df_db)

    # Sort to ensure row order doesn’t interfere with equality check
    result_df_sorted = result_df.sort_values(by=[0, 1, 2]).reset_index(drop=True)
    expected_df_sorted = expected_df.sort_values(by=[0, 1, 2]).reset_index(drop=True)

    assert result_df_sorted.equals(expected_df_sorted)


@pytest.mark.local
def test_complete_data_from_db():
    df_db = pd.read_csv(
        "/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv",
        header=None,
    )
    df = pd.read_csv(
        "/home/fatemeh/Downloads/bird/data/final/proc/m_data_index.csv", header=None
    )

    df_comp = complete_data_from_db(df, df_db)

    a = df_comp[df_comp[3] != -1]
    a = a.sort_values([0, 1, 2]).reset_index(drop=True)
    b = df.sort_values([0, 1, 2]).reset_index(drop=True)
    assert a.equals(b)


def test_accept_rule_2():
    data = {3: [-1] * 9 + [1] * 4 + [2] * 7}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=2)

    assert result is not None
    assert all(result[3] == 2)


def test_reject_rule_2_not_enough_l2():
    data = {3: [-1] * 9 + [1] * 4 + [2] * 3}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=2)

    assert result is None


def test_accept_rule_1():
    data = {3: [-1] * 9 + [1] * 5 + [2] * 3}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=1)

    assert result is not None
    assert all(result[3] == 1)


def test_reject_more_than_3_unique_labels():
    data = {3: [-1] * 3 + [1] * 3 + [2] * 3 + [3] * 3}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=2)

    assert result is None


def test_reject_only_negative_ones():
    data = {3: [-1] * 10}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=1)

    assert result is None


def test_accept_two_labels_with_negative():
    data = {3: [-1] * 6 + [1] * 5}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=1)

    assert result is not None
    assert all(result[3] == 1)


def test_reject_two_labels_with_negative_not_enough():
    data = {3: [-1] * 5 + [1] * 3}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=1)

    assert result is None


def test_reject_three_labels_with_negative_not_enough_l2():
    data = {3: [-1] * 5 + [1] * 6 + [2] * 2}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=2)

    assert result is None


def test_reject_single_label_not_enough():
    data = {3: [1] * 4}
    df = pd.DataFrame(data)
    result = evaluate_and_modify_df(df.copy(), rule=1)
    assert result is None
