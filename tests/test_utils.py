import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
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

    assert bu.equal_dataframe(df1, df2) is True


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

    assert bu.equal_dataframe(df1, df2) is False
