import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import behavior.data as bd
import behavior.data_processing as bdp


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


def test_change_format_json_file(tmp_path, sample_json_data):
    expected = pd.DataFrame(
        [
            [123, "2021-01-01 00:00:00", -1, 0, 0.1, 0.2, 0.3, 10.1],
            [123, "2021-01-01 00:00:00", -1, 0, 0.4, 0.5, 0.6, 10.2],
        ]
    )
    json_file = tmp_path / "input.json"
    with open(json_file, "w") as f:
        json.dump(sample_json_data, f)

    df = bdp.change_format_json_file(json_file)

    assert len(df) == 2  # 2 measurements
    assert expected.equals(df)


def test_change_format_json_files(tmp_path, sample_json_data):
    json_file1 = tmp_path / "validation_set.json"
    json_file2 = tmp_path / "train_set.json"
    json_file3 = tmp_path / "test_set.json"
    save_file = tmp_path / "output.csv"
    with open(json_file1, "w") as f:
        json.dump(sample_json_data, f)
    with open(json_file2, "w") as f:
        json.dump(sample_json_data, f)
    with open(json_file3, "w") as f:
        json.dump(sample_json_data, f)

    df = bdp.change_format_json_files(tmp_path, save_file)

    assert save_file.exists()
    content = save_file.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 6
    assert lines[0].startswith("123,2021-01-01 00:00:00,-1,0,0.100000")


@pytest.mark.local
def test_change_format_mat_file():
    data_file = "/home/fatemeh/Downloads/bird/data/data_from_Susanne/AnnAcc6016_20150501_110811-20150501_113058.mat"
    df = bdp.change_format_mat_file(data_file)

    expected = {(30, 52): 6, (61, 81): 10, (108, 128): 16, (141, 199): 5}
    dt = 6016, "2015-05-01 11:18:35"
    assert expected == bdp.get_start_end_inds(df, dt)

    expected = {(32, 105): 15, (106, 199): 5}
    dt = 6016, "2015-05-01 11:17:30"
    assert expected == bdp.get_start_end_inds(df, dt)


@pytest.mark.local
def test_change_format_mat_files():
    data_path = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
    df = bdp.change_format_mat_files(data_path)
    assert len(df) == 28213


@pytest.mark.local
def test_change_format_csv_file():
    data_file = "/home/fatemeh/Downloads/bird/data/data_from_Willem/AnM534_20120608_20120609.csv"
    df = bdp.change_format_csv_file(data_file)

    expected = {(0, 23): 9, (24, 59): 5}
    dt = 534, "2012-06-08 04:24:26"
    assert expected == bdp.get_start_end_inds(df, dt)


def test_build_index_n1():
    keys = [[1], [2], [3]]
    expected = {((1,),): [0], ((2,),): [1], ((3,),): [2]}
    assert bdp.build_index(keys, n_rows=1) == expected


def test_build_index_n2():
    keys = [[1], [2], [3]]
    expected = {((1,), (2,)): [0], ((2,), (3,)): [1]}
    assert bdp.build_index(keys, n_rows=2) == expected


def test_build_index_n3():
    keys = [[1], [2], [3]]
    expected = {((1,), (2,), (3,)): [0]}
    assert bdp.build_index(keys, n_rows=3) == expected


def test_build_index_too_few_elements():
    keys = [[1]]
    expected = {}
    assert bdp.build_index(keys, n_rows=2) == expected


def test_build_index_exact_elements():
    keys = [[1], [2]]
    expected = {((1,), (2,)): [0]}
    assert bdp.build_index(keys, n_rows=2) == expected


def test_build_index_duplicates_n1():
    keys = [[1], [1], [1]]
    expected = {((1,),): [0, 1, 2]}
    assert bdp.build_index(keys, n_rows=1) == expected


def test_build_index_duplicates_n2():
    keys = [[1], [1], [1]]
    expected = {((1,), (1,)): [0, 1]}
    assert bdp.build_index(keys, n_rows=2) == expected


def test_add_index():
    df_csv = """
6073,2016-06-07 12:34:34,-1,5,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,-1,5,0.186180,0.038160,0.921492,0.044410
6073,2016-06-07 12:34:34,-1,5,0.367562,0.042074,1.061825,0.044410"""

    df_db_csv = """
6073,2016-06-07 12:34:34,8,-1,0.300384,0.037182,1.015702,0.044410
6073,2016-06-07 12:34:34,9,-1,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,10,-1,0.186180,0.038160,0.921492,0.044410
6073,2016-06-07 12:34:34,11,-1,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,12,-1,0.093090,-0.208415,0.884200,0.044410"""

    expected_df_csv = """
6073,2016-06-07 12:34:34,9,5,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,10,5,0.186180,0.038160,0.921492,0.044410
6073,2016-06-07 12:34:34,11,5,0.367562,0.042074,1.061825,0.044410"""

    df = pd.read_csv(StringIO(df_csv), header=None)
    df_db = pd.read_csv(StringIO(df_db_csv), header=None)
    expected_df = pd.read_csv(StringIO(expected_df_csv), header=None)

    # tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file = tempfile.NamedTemporaryFile()
    bdp.add_index(df_db, df, tmp_file.name)
    result_df = pd.read_csv(tmp_file.name, header=None)

    assert result_df.equals(expected_df)


@pytest.mark.local
def test_map0():
    path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
    for name in ["s", "m", "w"]:
        df1 = pd.read_csv(path / f"{name}_index.csv", header=None)
        df2 = pd.read_csv(path / f"{name}_map0.csv", header=None)
        assert df1.equals(df2)


@pytest.mark.local
def test_correct_mistakes():
    dt = (534, "2012-06-08 12:39:39")
    path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
    for name in ["s", "j", "w", "m"]:
        df1 = pd.read_csv(path / f"{name}_map0.csv", header=None)
        df2 = bdp.correct_mistakes(df1, name)
        if name == "m":
            df1 = df1.sort_values([0, 1, 2]).reset_index(drop=True)
            assert df1.equals(df2)
        else:
            crop1 = df1[(df1[0] == dt[0]) & (df1[1] == dt[1])]
            crop2 = df2[(df2[0] == dt[0]) & (df2[1] == dt[1])]
            assert len(crop1) != 0
            assert len(crop2) == 0


def test_drop_groups_with_all_neg1():
    # fmt:off
    df = pd.DataFrame([
        [533,"2012-05-15 03:10:11",5,-1,   -0.519600,-0.105400,0.880500,0.020000],
        [533,"2012-05-15 03:10:11",6,-1,   -0.504800,-0.041900,0.880500,0.020000],
        [533,"2012-05-15 03:10:11",7,-1,   -0.391704, 0.071453,1.172469,0.020033],
        [533,"2012-05-15 03:10:11",8,-1,   -0.394823,-0.244503,0.856317,0.020033],
        [533,"2012-05-15 03:10:11",9,-1,   -0.055668, 0.134946,0.904723,0.020033]
    ])
    # fmt:on
    filtered = bdp.drop_groups_with_all_neg1(df)
    assert len(filtered) == 0


def test_complete_data_from_db_with_example():
    df_csv = """
6073,2016-06-07 12:34:34,9,11,0.367562,0.042074,1.061825,0.044410
6073,2016-06-07 12:34:34,11,5,0.586372,0.191781,1.155054,0.044410
6073,2016-06-07 12:38:39,0,10,0.357965,-0.109589,1.020608,0.004212
6073,2016-06-07 12:38:39,1,10,0.324376,0.019569,0.933268,0.004212
6073,2016-06-07 12:38:39,2,10,0.228407,-0.244618,0.868499,0.004212"""

    df_db_csv = """
6073,2016-06-07 12:34:34,8,-1,0.300384,0.037182,1.015702,0.044410
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

    expected_df_csv = """
6073,2016-06-07 12:34:34,8,-1,0.300384,0.037182,1.015702,0.044410
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

    result_df = bdp.complete_data_from_db(df, df_db)

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
        "/home/fatemeh/Downloads/bird/data/final/proc2/m_map.csv", header=None
    )

    df = bdp.drop_groups_with_all_neg1(df)
    df_comp = bdp.complete_data_from_db(df, df_db)

    df_comp = df_comp[df_comp[3] != -1]
    df = df[df[3] != -1]
    df_comp = df_comp.sort_values([0, 1, 2]).reset_index(drop=True)
    df = df.sort_values([0, 1, 2]).reset_index(drop=True)
    assert df.equals(df_comp)


def test_merge_prefer_valid():
    # fmt:off
    df1 = pd.DataFrame([
        [533,"2012-05-15 03:10:11",5,-1,   -0.519600,-0.105400,0.880500,0.020000],
        [533,"2012-05-15 03:10:11",6,-1,   -0.504800,-0.041900,0.880500,0.020000],
        [533,"2012-05-15 03:10:11",7, 6,   -0.391704, 0.071453,1.172469,0.020033],
        [533,"2012-05-15 03:10:11",8, 6,   -0.394823,-0.244503,0.856317,0.020033],
        [533,"2012-05-15 03:10:11",9, 6,   -0.055668, 0.134946,0.904723,0.020033]
    ])

    df2 = pd.DataFrame([
        [533,"2012-05-15 03:10:11",5,6,   -0.519570,-0.105422,0.880520,0.020033],
        [533,"2012-05-15 03:10:11",6,6,   -0.504756,-0.041928,0.880520,0.020033],
        [533,"2012-05-15 03:10:11",7,6,   -0.391704, 0.071453,1.172469,0.020033],
        [533,"2012-05-15 03:10:11",8,6,   -0.394823,-0.244503,0.856317,0.020033],
        [533,"2012-05-15 03:10:11",9,6,   -0.055668, 0.134946,0.904723,0.020033]
    ])
    # fmt:on
    merged = bdp.merge_prefer_valid(df1, df2)
    assert merged.equals(df2)


def test_accept_rule_2():
    data = {3: [-1] * 9 + [1] * 4 + [2] * 7}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=2)

    assert result is not None
    assert all(result[3] == 2)


def test_reject_rule_2_not_enough_l2():
    data = {3: [-1] * 9 + [1] * 4 + [2] * 3}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=2)

    assert result is None


def test_accept_rule_1():
    data = {3: [-1] * 9 + [1] * 5 + [2] * 3}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=1)

    assert result is not None
    assert all(result[3] == 1)


def test_reject_more_than_3_unique_labels():
    data = {3: [-1] * 3 + [1] * 3 + [2] * 3 + [3] * 3}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=2)

    assert result is None


def test_reject_only_negative_ones():
    data = {3: [-1] * 10}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=1)

    assert result is None


def test_accept_two_labels_with_negative():
    data = {3: [-1] * 6 + [1] * 5}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=1)

    assert result is not None
    assert all(result[3] == 1)


def test_reject_two_labels_with_negative_not_enough():
    data = {3: [-1] * 5 + [1] * 3}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=1)

    assert result is None


def test_accept_based_on_rule_order_label_matter():
    data = {3: [-1] * 5 + [4] * 6 + [3] * 7}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=1)

    assert result is not None
    assert all(result[3] == 4)

    result = bdp.evaluate_and_modify_df(df.copy(), rule=2)

    assert result is not None
    assert all(result[3] == 3)


def test_reject_three_labels_with_negative_not_enough_l2():
    data = {3: [-1] * 5 + [1] * 6 + [2] * 2}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=2)

    assert result is None


def test_reject_single_label_not_enough():
    data = {3: [1] * 4}
    df = pd.DataFrame(data)
    result = bdp.evaluate_and_modify_df(df.copy(), rule=1)
    assert result is None


def test_process_moving_window():
    rule_df = bdp.get_rules().rule_df
    ind2name = bdp.get_rules().ind2name
    glen = 6

    # fmt:off
    df = pd.DataFrame([
    [533, "2012-05-15 03:10:11", 0, 6, -0.184313, -0.280029, 0.929683, 0.020033],
    [533, "2012-05-15 03:10:11", 1, 6, -0.170279, -0.252818, 0.929683, 0.020033],
    [533, "2012-05-15 03:10:11", 2, 6, -0.199906, -0.274738, 0.929683, 0.020033],
    [533, "2012-05-15 03:10:11", 3, 5, -0.222517, -0.222583, 0.904723, 0.020033],
    [533, "2012-05-15 03:10:11", 4, 6, -0.337907, -0.205953, 0.856317, 0.020033],
    [533, "2012-05-15 03:10:11", 5, 6, -0.519570, -0.105422, 0.880520, 0.020033],
    [533, "2012-05-15 03:10:11", 6, 6, -0.504756, -0.041928, 0.880520, 0.020033]
    ])
    # fmt:on

    new_df = bdp.process_moving_window_given_dt(df, rule_df, ind2name, glen)
    df.iloc[3, 3] = 6  # just for the assert
    assert new_df[0].equals(df.iloc[:glen])
    assert new_df[1].equals(df.iloc[1 : 1 + glen])
    assert len(new_df) == 2


@pytest.mark.local
def test_process_moving_window_given_dt():
    rule_df = bdp.get_rules().rule_df
    ind2name = bdp.get_rules().ind2name

    glen = 20  # group length
    dt = 6011, "2015-04-30 09:10:57"

    expected = pd.read_csv(
        f"/home/fatemeh/Downloads/bird/data/final/proc/example_moving_win_{dt[0]}_{dt[1]}.csv",
        header=None,
    )
    cut = pd.read_csv(
        f"/home/fatemeh/Downloads/bird/data/final/proc/example_{dt[0]}_{dt[1]}.csv",
        header=None,
    )

    cut_dt = cut[(cut[0] == dt[0]) & (cut[1] == dt[1])].copy()
    new_df = bdp.process_moving_window_given_dt(cut_dt, rule_df, ind2name, glen)
    new_df = pd.concat(new_df)
    new_df.equals(expected)
    """
    # The data is generated as below and manually checked

    glen = 20  # group length
    dt = 6011, "2015-04-30 09:10:57"
    df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc/m_data_complete.csv", header=None)
    
    rule_df = get_rules().rule_df
    ind2name = get_rules().ind2name
    ignore_labels = get_rules().ignore_labels
    df_unq_labels = np.unique(df[3])
    mapping = {idx: -1 if idx in ignore_labels else idx for idx in df_unq_labels}
    df[3] = df[3].map(mapping)
    cut = df[(df[0] == dt[0]) & (df[1] == dt[1])]
    
    cut_dt = cut[(cut[0] == dt[0]) & (cut[1] == dt[1])].copy()
    new_df = bdp.process_moving_window_given_dt(cut_dt, rule_df, ind2name, glen)
    new_df = pd.concat(new_df)

    cut.to_csv(f"/home/fatemeh/Downloads/bird/data/final/proc/example_{dt[0]}_{dt[1]}.csv", index=False, header=None, float_format="%.6f")
    new_df.to_csv(f"/home/fatemeh/Downloads/bird/data/final/proc/example_moving_win_{dt[0]}_{dt[1]}.csv", index=False, header=None, float_format="%.6f")
    """


@pytest.mark.local
def test_shift_df():

    # test for shift two device and time
    glen = 20  # group length
    dts = [
        [6011, "2015-04-30 09:10:57"],
        [6011, "2015-04-30 09:10:44"],
    ]

    expected = pd.read_csv(
        f"/home/fatemeh/Downloads/bird/data/final/proc2/test_shift_2.csv",
        header=None,
    )
    df = pd.read_csv(
        f"/home/fatemeh/Downloads/bird/data/final/proc2/combined.csv",
        header=None,
    )

    new_df = bdp.shift_df(df, glen, dts)
    new_df.equals(expected)

    bdp.check_batches(new_df, batch_size=glen)

    # Test for all shifted data
    df = pd.read_csv(
        f"/home/fatemeh/Downloads/bird/data/final/proc2/shift.csv",
        header=None,
    )
    bdp.check_batches(df, batch_size=glen)


@pytest.mark.local
def test_combine_shift():
    dfc = pd.read_csv(
        "/home/fatemeh/Downloads/bird/data/final/proc2/combined.csv", header=None
    )
    dfs = pd.read_csv(
        "/home/fatemeh/Downloads/bird/data/final/proc2/shift.csv", header=None
    )
    gc = dfc.groupby([0, 1])
    gs = dfs.groupby([0, 1])
    cdt = set(gc.groups.keys())
    sdt = set(gs.groups.keys())

    assert len(sdt.difference(cdt)) == 0

    diff_dts = cdt.difference(sdt)
    for dt in diff_dts:
        slice = gc.get_group(dt)
        # Not a srong assert
        assert len(slice[slice[3] != -1]) < 20


def test_drop_duplicates():
    # fmt:off
    df = pd.DataFrame([
        [533,"2012-05-15 03:10:11",5,6,   -0.519570,-0.105422,0.880520,0.020033],
        [533,"2012-05-15 03:10:11",6,6,   -0.504756,-0.041928,0.880520,0.020033],
        [533,"2012-05-15 03:10:11",7,6,   -0.391704, 0.071453,1.172469,0.020033],
        [533,"2012-05-15 03:10:11",8,6,   -0.394823,-0.244503,0.856317,0.020033],
    ])
    # fmt:on
    df2 = pd.concat([df, df]).reset_index(drop=True)

    result = bdp.drop_duplicates(df2, glen=4)
    assert result.equals(df)


@pytest.mark.local
def test_get_start_end_inds():
    # fmt:off
    df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/combined.csv", header=None)
    dt = 6011, "2015-04-30 09:10:31"
    expected = {(0, 13): 9, (43, 60): 9, (84, 104): 9, (114, 132): 5, (165, 199): 9}
    # fmt:on
    assert expected == bdp.get_start_end_inds(df, dt)


@pytest.mark.local
def test_change_format_csv_files():
    data_path = Path("/home/fatemeh/Downloads/bird/data/data_from_Willem")
    df = bdp.change_format_csv_files(data_path)
    assert len(df) == 67689


@pytest.mark.local
@pytest.mark.debug
def test_issue_previous_m_data_format():
    df = pd.read_csv(
        "/home/fatemeh/Downloads/bird/data/final/proc2/m_complete.csv", header=None
    )
    df = df.sort_values(by=[0, 1], ignore_index=True)
    groups = df.groupby([0, 1])
    issues = dict()
    for name, group in groups:
        if (
            group.iloc[0, 3] == -1
            and len(group[group[3] != -1]) > 20
            and group[group[3] != -1].iloc[0, 2] > 4
        ):
            print(
                f"{name[0]},{name[1]}, {len(group[group[3]!=-1])}, {group[group[3]!=-1].iloc[0,2]}"
            )
            issues[name] = len(group[group[3] != -1])
    issues = dict(sorted(issues.items(), key=lambda x: x[1]))
    max_wrong_labels = sum([(round(i / 20) * 20) // 20 for i in issues.values()])
    assert max_wrong_labels == 52
    assert len(issues) == 15


@pytest.mark.local
@pytest.mark.debug
def test_labels_comes_together():
    """
    Check which labels come together.
    """
    with open(
        "/home/fatemeh/Downloads/bird/data/final/proc2/combined_label_range.txt", "r"
    ) as file:
        lines = file.readlines()
        data = []
        for line in lines:
            parts = line.strip().split(",")
            labels = [int(part.split(":")[0]) for part in parts[2:]]
            labels = np.unique(labels)
            if len(labels) > 1:
                labels = tuple(sorted(labels))
                data.append(labels)
    print(Counter(data))
    # {(2, 8): 36, (5, 9): 25, (6, 9): 22, (5, 6): 19, (0, 2): 14, (0, 8): 8, (0, 1): 7, (1, 2): 3, (1, 8): 3, (0, 2, 8): 1, (1, 5): 1}


def test_get_rules():
    rules = bdp.get_rules().rule_df
    assert rules.loc["Pecking", "SitStand"] == 1
    assert rules.loc["SitStand", "Pecking"] == 2


def test_map_to_nearest_divisible_20():
    assert bdp.map_to_nearest_divisible_20(2, 18) == [0, 20]
    assert bdp.map_to_nearest_divisible_20(23, 37) == [20, 40]
    assert bdp.map_to_nearest_divisible_20(35, 55) == [40, 60]
    assert bdp.map_to_nearest_divisible_20(30, 50) == [40, 60]
