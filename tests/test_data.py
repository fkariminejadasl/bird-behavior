from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

import behavior.data as bd


def test_select_random_groups():
    # Sample data
    data = {
        0: [541, 541, 606, 606, 805, 805],
        1: pd.to_datetime(
            [
                "2012-05-18 02:58:35",
                "2012-05-18 02:58:35",
                "2014-05-24 11:29:37",
                "2014-05-24 11:29:37",
                "2014-06-07 09:45:20",
                "2014-06-07 09:45:20",
            ]
        ),
        2: [0, 1, 20, 21, 0, 1],
        3: [9, 9, 9, 9, 9, 9],
        4: [-0.365094, -0.390851, -0.474436, -0.299248, 0.317241, -0.092720],
        5: [-0.126978, -0.184958, 0.179352, 0.040693, -0.108314, -0.263921],
        6: [0.917604, 0.893047, 0.951916, 0.855748, 1.119908, 0.922367],
        7: [0.423645, 0.423645, 0.241683, 0.241683, 0.787784, 0.787784],
    }
    a = pd.DataFrame(data)
    a.index = [240, 241, 1900, 1901, 4180, 4181]

    np.random.seed(123)
    sel_df = bd.select_random_groups(a, n_samples=2, glen=2)

    # Assert expected rows based on seed
    # For replace=True, the seed is 123
    # expected_sel_df = pd.concat([a.loc[4180:4181], a.loc[1900:1901]])
    expected_sel_df = pd.concat([a.loc[240:241], a.loc[1900:1901]])

    pd.testing.assert_frame_equal(sel_df, expected_sel_df)


def test_create_balanced_data():
    # def test_create_balanced_data(tmp_path):
    # Simulate a CSV file
    data = {
        0: [1] * 12 + [2] * 12,
        1: pd.date_range("2020-01-01", periods=24, freq="min"),
        2: list(range(12)) + list(range(12)),
        3: [0] * 12 + [9] * 12,  # Two classes: 0 and 9
        4: np.random.rand(24),
        5: np.random.rand(24),
        6: np.random.rand(24),
        7: np.random.rand(24),
    }
    df = pd.DataFrame(data)

    # Set seed for reproducibility
    np.random.seed(42)

    # Test function
    glen = 2
    result_df = bd.create_balanced_data(df, keep_labels=[0, 9], n_samples=2, glen=glen)

    # Check size: 2 samples * 2 rows (glen=2) * 2 labels = 8 rows
    assert len(result_df) == 8

    # Check class distribution
    class_counts = result_df[3].value_counts().to_dict()
    assert class_counts == {0: 4, 9: 4}

    # Check consecutive values in column 2
    values = result_df[2].values.reshape(-1, glen)
    assert np.all(np.diff(values, axis=1) == 1)

    assert set(result_df.columns) == set(range(8))

    # Check size: 12/glen=6 samples * 2 rows (glen=2) * 2 labels = 24 rows
    result_df = bd.create_balanced_data(df, keep_labels=[0, 9], glen=glen)
    assert len(result_df) == 24
    print("Done")


def test_identify_and_process_groups():
    # fmt: off
    data = [
        [1, 20], [2, 14], [1, 50], [2, 34], [3, 28], [4, 22], [5, 18], [6, 15], [7, 14], [8, 13],
        [9, 12], [10, 11], [11, 10], [12, 9], [13, 8], [14, 7], [15, 6], [16, 5], [17, 4], [18, 3],
        [19, 2], [20, 1], [21, 50], [22, 49], [23, 48], [24, 47], [25, 46], [26, 45], [27, 44],
        [28, 43], [29, 42], [30, 41], [31, 40], [32, 39], [33, 38], [34, 37], [35, 36], [36, 35],
        [37, 34], [38, 33], [39, 32], [40, 31], [41, 30], [42, 29], [43, 28], [44, 27], [45, 26], [46, 25]
    ]
    # fmt: on
    processed_groups = bd.identify_and_process_groups(data)
    np.testing.assert_equal(np.array(data)[2:22], np.array(processed_groups[0]))
    np.testing.assert_equal(np.array(data)[22:42], np.array(processed_groups[1]))


@pytest.mark.ignore
def test_get_data():
    device_id = 534
    start_time = "2012-06-08 10:28:58"
    database_url = "postgresql://username:yourpass@pub.e-ecology.nl:5432/eecology"
    gimus, idts, llat = bd.get_data(database_url, device_id, start_time, start_time)
    assert gimus.shape == (60, 4)


@pytest.mark.local
@pytest.mark.parametrize(
    "file_path",
    [
        "/home/fatemeh/Downloads/bird/data/final/s_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/j_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/w_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/m_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/corrected_combined_unique_sorted012.csv",
        "/home/fatemeh/Downloads/bird/data/final/orig/s_data_orig_with_index.csv",
        "/home/fatemeh/Downloads/bird/data/final/s_data_shift/s_data_balanced_0.csv",
        "/home/fatemeh/Downloads/bird/data/final/s_data_shift/s_data_unbalanced.csv",
    ],
)
def test_data_contents(file_path):
    df = pd.read_csv(file_path, header=None)

    # Test groups of 20
    g = df.groupby([0, 1])
    s = g.size()
    assert np.array_equal(np.unique(s) % 20, np.zeros_like(np.unique(s)))

    # Test unique labels
    g1 = df.groupby([0, 1, 3])
    s1 = g1.size()
    assert np.array_equal(np.unique(s1) % 20, np.zeros_like(np.unique(s1)))

    # Test unique GPS speeds
    g2 = df.groupby([0, 1, 3, 7])
    s2 = g2.size()
    assert np.array_equal(np.unique(s2) % 20, np.zeros_like(np.unique(s2)))

    # # Check indices every 20 rows are divisible by 20
    # uniq_inds = np.unique(df.iloc[::20, 2])
    # assert np.array_equal(uniq_inds % 20, np.zeros_like(uniq_inds))

    # Check consecutive values in column 2
    values = df[2].values.reshape(-1, 20)
    assert np.all(np.diff(values, axis=1) == 1)

    # Verify there are no duplicate rows
    assert len(df[df.duplicated() == True]) == 0


@pytest.mark.debug
def test_find_index_jumps(df):
    """
    Identify discontinuities in the indices. Determine whether the signal contains
    labeled segments separated by unlabeled intervals.
    """
    df = df.sort_values([0, 1, 2])
    groups = df.groupby([0, 1])
    for k, v in groups:
        diff = np.diff(v[2].values)
        idxs = np.where(diff != 1)[0]
        large_diff = [diff[i] for i in idxs]
        if len(large_diff) != 0:
            print(k, large_diff)
    # no jupms in m_data_index.csv, w_data_index.csv
    # s_data_index.csv there are jumps but OK
    # j_data_index.csv
    # (6080, '2014-06-26 07:52:46') [11]
    # (6080, '2014-06-26 07:55:34') [8]
    # (6080, '2014-06-26 07:56:29') [8]
    # (6080, '2014-06-26 07:59:49') [21]
    # (6080, '2014-06-26 08:02:19') [7]
    # (6080, '2014-06-26 07:59:49') [21] was wrong. But if we ignore label 10, everything is OK.

    # from pathlib import Path
    # path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
    # for i in ["s", "j", "m", "w"]:
    #     df = pd.read_csv(path / f"{i}_map.csv", header=None)
    #     print(f"Testing {i} data")
    #     test_find_index_jumps(df)
    # print("Done")
