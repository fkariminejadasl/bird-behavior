import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib.figure import Figure

from behavior import utils as bu
from behavior.utils import plot_all


# Mock data generation
def generate_mock_data():
    np.random.seed(42)

    # Creating mock dataframe_db (full database)
    num_samples = 100
    device_id = 6011
    timestamp = "2015-04-30 09:09:26"
    indices = np.arange(num_samples)
    imu_data = np.random.randn(num_samples, 3)  # Simulated IMU data
    labels = np.random.randint(0, 5, size=num_samples)

    dataframe_db = pd.DataFrame(
        {
            0: device_id,  # Device ID
            1: timestamp,  # Timestamp
            2: indices,  # IMU indices
            3: labels,  # Labeling
            4: imu_data[:, 0],
            5: imu_data[:, 1],
            6: imu_data[:, 2],
            7: np.random.uniform(30, 50, size=num_samples),  # Simulated GPS info
        }
    )

    # Creating mock dataframe (subset of dataframe_db)
    labeled_indices = np.sort(np.random.choice(indices, size=50, replace=False))
    dataframe = dataframe_db[dataframe_db[2].isin(labeled_indices)].copy()

    return dataframe, dataframe_db


@pytest.fixture
def mock_data():
    return generate_mock_data()


def test_plot_all_runs(mock_data):
    """Test if plot_all runs without errors and returns a figure."""
    dataframe, dataframe_db = mock_data
    fig = plot_all(dataframe, dataframe_db, glen=10)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_all_correct_axes_count(mock_data):
    """Test if the function produces the correct number of subplots."""
    dataframe, dataframe_db = mock_data
    fig = plot_all(dataframe, dataframe_db, glen=10)
    assert len(fig.axes) == 2  # Expecting two rows of plots
    plt.close(fig)


def test_plot_all_vertical_lines(mock_data):
    """Test if the vertical lines are placed correctly."""
    dataframe, dataframe_db = mock_data
    fig = plot_all(dataframe, dataframe_db, glen=10)
    ax = fig.axes[0]  # Checking the first subplot

    # Extract x-coordinates of vertical lines
    vertical_lines = [
        line.get_xdata()[0] for line in ax.get_lines() if line.get_linestyle() == "-"
    ]
    start_indices = dataframe[2].values[::10]

    for start in start_indices:
        assert start in vertical_lines

    plt.close(fig)


def test_equal_dataframes_true():
    df1 = pd.DataFrame(
        [
            [1, "2023-01-01 00:00:00", 10, 1, 0.123456, 0.234567],
            [1, "2023-01-01 00:00:00", 11, 1, 0.333333, 0.444444],
        ]
    )

    df2 = pd.DataFrame(
        [
            [1, "2023-01-01 00:00:00", 11, 1, 0.333334, 0.444443],
            [1, "2023-01-01 00:00:00", 10, 1, 0.123457, 0.234566],
        ]
    )

    assert bu.equal_dataframe(df1, df2, [0, 1, 2, 3, 4, 5]) is True

    df2 = pd.DataFrame(
        [
            [1, "2023-01-01 00:00:00", -1, 1, 0.333334, 0.444443],
            [1, "2023-01-01 00:00:00", -1, 1, 0.123457, 0.234566],
        ]
    )

    assert bu.equal_dataframe(df1, df2, [0, 1, 3, 4, 5]) is True


def test_equal_dataframes_false():
    df1 = pd.DataFrame(
        [
            [1, "2023-01-01 00:00:00", 10, 1, 0.1234, 0.2345],
        ]
    )
    df2 = pd.DataFrame(
        [
            [1, "2023-01-01 00:00:00", 10, 1, 0.1234, 0.9999],
        ]
    )

    assert bu.equal_dataframe(df1, df2, [0, 1, 2, 3, 4, 5]) is False


def test_equal_csv_files_ignores_row_order_and_resets_index(tmp_path):
    df1 = pd.DataFrame(
        [
            [
                6210,
                "2016-05-09 10:27:20",
                18,
                -1,
                0.502449,
                -0.210526,
                0.882411,
                0.074034,
            ],
            [
                6210,
                "2016-05-09 10:27:20",
                17,
                -1,
                0.541626,
                -0.151072,
                0.888340,
                0.074034,
            ],
        ]
    )
    df2 = df1.iloc[[1, 0]].copy()

    file1 = tmp_path / "a.csv"
    file2 = tmp_path / "b.csv"
    df1.to_csv(file1, index=False, header=None)
    df2.to_csv(file2, index=False, header=None)

    assert bu.equal_csv_files(
        file1,
        file2,
        cols_to_compare=list(range(8)),
        sort_cols=[0, 1, 2],
        round_cols=[4, 5, 6, 7],
        precision=6,
    )


@pytest.mark.local
@pytest.mark.parametrize(
    "file_path1, file_path2",
    [
        (
            "/home/fatemeh/Downloads/bird/data/final/m_format.csv",
            "/home/fatemeh/Downloads/bird/data/final/m_index.csv",
        ),
        (
            "/home/fatemeh/Downloads/bird/data/final/w_format.csv",
            "/home/fatemeh/Downloads/bird/data/final/w_index.csv",
        ),
    ],
)
def test_files_equal(file_path1, file_path2):
    df1 = pd.read_csv(file_path1, header=None)
    df2 = pd.read_csv(file_path2, header=None)

    assert df1.equals(df2)
    assert bu.equal_dataframe(df1, df2, [0, 1, 2, 3, 4, 5, 6, 7])


def test_full_coverage_without_drops():
    labels = torch.tensor([0] * 5 + [1] * 7)
    a_idx, b_idx = bu.stratified_split(labels, split_ratios=[0.9, 0.1], seed=0)
    covered = torch.cat([a_idx, b_idx])
    # Expect coverage to equal all indices
    assert covered.numel() == labels.numel()
    assert torch.equal(torch.sort(covered).values, torch.arange(labels.numel()))


def test_stratified_split():
    # class sizes not multiples of ratios: remainder will be dropped by current logic
    labels = torch.tensor([0] * 5 + [1] * 7 + [2] * 11)
    ratios = [0.9, 0.1]
    a_idx, b_idx = bu.stratified_split(labels, split_ratios=ratios, seed=0)

    # fmt: off
    assert labels[a_idx].equal(torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    assert labels[b_idx].equal(torch.tensor([0, 1, 2, 2]))
    # fmt: on
