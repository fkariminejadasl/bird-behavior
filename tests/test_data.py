import numpy as np
import pandas as pd
import pytest

import behavior.data as bd


@pytest.mark.ignore
def test_get_data():
    device_id = 534
    start_time = "2012-06-08 10:28:58"
    database_url = "postgresql://username:yourpass@pub.e-ecology.nl:5432/eecology"
    gimus, idts, llat = bd.get_data(database_url, device_id, start_time, start_time)
    assert gimus.shape == (60, 4)


@pytest.mark.parametrize(
    "file_path",
    [
        "/home/fatemeh/Downloads/bird/data/final/s_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/j_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/w_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/m_data.csv",
        "/home/fatemeh/Downloads/bird/data/final/corrected_combined_unique_sorted012.csv",
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

    # Check indices every 20 rows are divisible by 20
    uniq_inds = np.unique(df.iloc[::20, 2])
    assert np.array_equal(uniq_inds % 20, np.zeros_like(uniq_inds))

    # Verify there are no duplicate rows
    assert len(df[df.duplicated() == True]) == 0


# df_s = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/s_data.csv", header=None)
# df_w = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/w_data.csv", header=None)
# df_j = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/j_data.csv", header=None)
# df_m = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/m_data.csv", header=None)
# g_s = df_s.groupby([0,1,3])
# g_w = df_w.groupby([0,1,3])
# g_j = df_j.groupby([0,1,3])
# g_m = df_m.groupby([0,1,3])
# gk_s = [k for k, _ in g_s]
# gk_w = [k for k, _ in g_w]
# gk_j = [k for k, _ in g_j]
# gk_m = [k for k, _ in g_m]
# len(gk_s)
# len(gk_w)
# len(set(gk_w).difference(set(gk_s)))
# len(set(gk_s).difference(set(gk_w)))

