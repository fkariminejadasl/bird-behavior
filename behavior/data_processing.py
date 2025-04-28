from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

import behavior.data as bd
import behavior.utils as bu


def change_format_json_file(json_file, save_file):
    """
    read json data and save it as formatted csv file

    input:  device, time, index, label, imux, imuy, imuz, gps
    e.g. row: 757,2014-05-18 06:58:26,20,0,-0.09648467,-0.04426107,0.45049885,8.89139205
    """

    all_measurements, ldts = bd.load_all_data_from_json(json_file)

    items = []
    for meas, ldt in tqdm(
        zip(all_measurements, ldts), total=len(ldts)
    ):  # N x {10,20} x 4
        # labels read 0-based
        label = ldt[0]
        device_id = ldt[1]
        timestamp = ldt[2]
        start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        for i in meas:
            i = np.round(i, 6)
            item = (
                f"{device_id},{start_time},{-1},{label},{i[0]:.6f},{i[1]:.6f},"
                f"{i[2]:.6f},{i[3]:.6f}\n"
            )
            items.append(item)

    with open(save_file, "w") as f:
        for item in items:
            f.write(item)


def change_format_mat_file(mat_file):
    dd = loadmat(mat_file)["outputStruct"]
    n_data = dd["nOfSamples"][0][0][0][0]

    rows = []
    for i in range(n_data):
        year, month, day, hour, min, sec = (
            dd["year"][0][0][0, i],
            dd["month"][0][0][0, i],
            dd["day"][0][0][0, i],
            dd["hour"][0][0][0, i],
            dd["min"][0][0][0, i],
            dd["sec"][0][0][i, 0],
        )
        t = datetime(year, month, day, hour, min, sec).strftime("%Y-%m-%d %H:%M:%S")

        device_id = dd["sampleID"][0][0][0, i]
        imu_x = dd["accX"][0][0][i][0][0]
        imu_y = dd["accY"][0][0][i][0][0]
        imu_z = dd["accZ"][0][0][i][0][0]
        gps_single = dd["gpsSpd"][0][0][i, 0]
        tags = dd["tags"][0][0][0][i]

        inds = np.where(tags[:, 1] == 1)[0]
        if len(inds) == 0:
            continue

        labels = tags[inds][:, 0] - 1  # 0-based
        for ind, label, x, y, z in zip(
            inds, labels, imu_x[inds], imu_y[inds], imu_z[inds]
        ):
            if any([np.isnan(x), np.isnan(y), np.isnan(z), np.isnan(gps_single)]):
                continue
            ig = np.round(np.array([x, y, z, gps_single]), 6)
            item = [device_id, t, ind, label, ig[0], ig[1], ig[2], ig[3]]
            rows.append(item)
    return pd.DataFrame(rows)


def change_format_mat_files(mat_path: Path, save_file=None) -> pd.DataFrame:
    all_rows = []
    for mat_file in tqdm(mat_path.glob("An*mat")):
        rows = change_format_mat_file(mat_file)
        all_rows.append(rows)
    df = pd.concat(all_rows)
    if save_file is not None:
        df.to_csv(save_file, index=False, header=None, float_format="%.6f")
    return df


def change_format_csv_file_pd(csv_file):
    # Same as change_format_csv_file but only with pandas
    # 1) parse "NaN" as actual NaN
    df = pd.read_csv(
        csv_file,
        sep=r",\s*",
        engine="python",
        names=["device_id", "start_time", "c2", "c3", "c4", "c5", "c6", "label_conf"],
        dtype=str,
        na_values=["NaN"],
    )

    # 2) split out label/conf, filter conf==0
    df[["label", "conf"]] = df["label_conf"].str.split(" ", expand=True)
    df["label"] = df["label"].astype(int) - 1
    df["conf"] = df["conf"].astype(int)
    df = df[df["conf"] != 0]

    # 3) drop any row with a true NaN in c3,c4,c5 or c6
    df = df.dropna(subset=["c3", "c4", "c5", "c6"])

    # 4) reformat datetime
    df["start_time"] = pd.to_datetime(
        df["start_time"], format="%m/%d/%Y %H:%M:%S"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 5) select & reorder columns
    return df[["device_id", "start_time", "c2", "label", "c3", "c4", "c5", "c6"]]


def change_format_csv_file(csv_file):
    rows = []
    with open(csv_file, "r") as f:
        for r in f:
            r = r.strip().split(", ")
            device_id = r[0]
            start_time = datetime.strptime(r[1], "%m/%d/%Y %H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            label, conf = r[-1].split(" ")
            label = int(label) - 1  # zero-based
            if conf == "0":
                continue
            if r[3] == "NaN" or r[4] == "NaN" or r[5] == "NaN" or r[6] == "NaN":
                continue
            rows.append([device_id, start_time, r[2], label, r[3], r[4], r[5], r[6]])
    df = pd.DataFrame(rows)
    return df


def change_format_csv_files(input_dir: Path, save_file=None):
    """
    Read & reformat many CSVs, then write one combined output.

    :param csv_files: list of input file paths
    :param save_file: path to write the combined CSV
    :returns: the concatenated DataFrame
    """
    dfs = []
    csv_files = list(input_dir.glob("*.csv"))
    for csv_path in tqdm(csv_files, total=len(csv_files)):
        df = change_format_csv_file(csv_path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if save_file is not None:
        combined.to_csv(save_file, index=False, header=None)

    return combined


def build_index(data: np.ndarray, n_rows=3, step=1):
    """
    Create a dictionary where key = tuple of n_rows consecutive tuples, value = starting index
    """
    index_map = {}
    for i in range(0, len(data) - n_rows + 1, step):
        key = tuple(tuple(data[i + j]) for j in range(n_rows))
        if key in index_map:
            index_map[key].append(i)
        else:
            index_map[key] = [i]
    return index_map


def match_forward_backward(index_maps, keys_rounded, i):
    """
    Try to find the best match at index i by checking forward and backward
    for n=3, 2, 1 in that order. Returns (match_index_in_db, start_row_in_df, n_rows)
    """
    max_n = 3
    L = len(keys_rounded)

    for n_rows in reversed(range(1, max_n + 1)):
        # Forward match
        if i + n_rows <= L:
            key_fwd = tuple(tuple(keys_rounded[i + j]) for j in range(n_rows))
            indices_fwd = index_maps[n_rows - 1].get(key_fwd)
            if indices_fwd and len(indices_fwd) == 1:
                return indices_fwd[0], i, n_rows

        # Backward match
        if i - (n_rows - 1) >= 0:
            key_bwd = tuple(tuple(keys_rounded[i - j]) for j in reversed(range(n_rows)))
            indices_bwd = index_maps[n_rows - 1].get(key_bwd)
            if indices_bwd and len(indices_bwd) == 1:
                return indices_bwd[0], i - (n_rows - 1), n_rows

    return None, None, None


def add_index(df_db, df, save_file):
    df_db = df_db.copy()
    df = df.copy()

    # Settings
    tol = 1e-4  # precision 6 is issue in original data
    precision = -int(np.log10(tol))

    # Preprocess keys
    df_db.iloc[:, [4, 5, 6, 7]] = np.round(df_db.iloc[:, [4, 5, 6, 7]], precision)
    keys_rounded = df_db.iloc[:, [0, 1, 4, 5, 6, 7]].values
    index_map1 = build_index(keys_rounded, 1)
    index_map2 = build_index(keys_rounded, 2)
    index_map3 = build_index(keys_rounded, 3)
    index_maps = [index_map1, index_map2, index_map3]

    # Pre-round df once
    df_values = df.copy()
    df_values.iloc[:, [4, 5, 6, 7]] = np.round(df.iloc[:, [4, 5, 6, 7]], precision)
    df_values = df_values.iloc[:, [0, 1, 4, 5, 6, 7]].values

    # Output buffer
    output_lines = []
    ind = 0
    j = 0
    pbar = tqdm(total=len(df))
    while j < len(df):
        sel_ind, start_j, n_row = match_forward_backward(index_maps, df_values, j)
        if sel_ind is None:
            j += 1
            pbar.update(1)
            continue  # No match found

        # Match found
        match_offset = j - start_j  # Position of current j in the matched window
        ind = sel_ind + match_offset
        i = df.iloc[j]
        imu_ind = df_db.iloc[ind, 2]
        item = (
            f"{i[0]},{i[1]},{imu_ind},{i[3]},{i[4]:.6f},{i[5]:.6f},"
            f"{i[6]:.6f},{i[7]:.6f}\n"
        )
        output_lines.append(item)

        j += 1
        pbar.update(1)

    # Write all at once
    with open(save_file, "w") as file:
        file.writelines(output_lines)


def correct_mistakes(df, name):
    """
    Removes mislabeled entries from s_data, w_data, and j_data based on identify_mistakes and manual checks.
    Entries are identified by (id, timestamp) and removed accordingly.
    """

    # fmt: off
    # Mistaken entries to remove
    remove_both = [
    (534, "2012-06-08 12:39:39"),
    (533, "2012-05-27 03:50:46"),
    (533, "2012-05-15 14:00:34"),
    (533, "2012-05-15 11:53:34"),
    (533, "2012-05-15 11:38:20"),
    (533, "2012-05-15 09:52:18"),
    (533, "2012-05-15 09:47:29"),
    (533, "2012-05-15 05:36:44"),
    (533, "2012-05-15 05:22:11"),
    ]
    remove_w = [
    (534, "2012-06-08 06:16:42"),
    (534, "2012-06-08 06:26:22"),
    (534, "2012-06-08 06:31:10"),
    (606, "2014-05-15 16:57:28"),
    ]
    remove_s = [(533, "2012-05-15 05:41:52")]
    remove_j = [
    (534, "2012-06-08 12:39:39"),
    (533, "2012-05-15 11:53:34"),
    ]

    # Helper to remove rows from a DataFrame based on (id, timestamp)
    def remove_entries(df, to_remove):
        to_remove_set = set(to_remove)
        return df[~df[[0, 1]].apply(tuple, axis=1).isin(to_remove_set)]

    df = df.sort_values([0, 1, 2]).reset_index(drop=True)
    
    if name == "s":
        df = remove_entries(df, remove_both + remove_s)
    if name == "w":
        df = remove_entries(df, remove_both + remove_w)
    if name == "j":
        df = remove_entries(df, remove_j)
    if name == "m":
        df = df

    return df


def map_new_labels(df, new2old_labels, save_file=None, ignore_labels=None):
    """
    Map new labels to old labels and remove data contining ignored labels
    Args:
        df (pd.DataFrame): DataFrame containing the labels to be mapped.
        new2old_labels (dict): Dictionary mapping new labels to old labels.
        ignore_labels (list): List of labels to ignore.
        save_file (str, optional): Path to save the modified DataFrame. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame with mapped labels and ignored labels removed.

    Example:
    ignore_labels = [7, 10, 13, 14, 15, 16]
    """
    df[3] = df[3].map(new2old_labels)
    # Keep rows where df[3] is NOT in ignore_labels
    if ignore_labels is not None:
        df = df[~df[3].isin(ignore_labels)]
    if save_file is not None:
        df.to_csv(save_file, index=False, header=None, float_format="%.6f")
    return df


def get_rules():
    """
    Get rules for label selection

    Example:
    rule["Flap"]["ExFlap"] is 2, which mean the second label ExFlap is selected.
    """
    # fmt: off
    labels = [
        "Flap", "ExFlap", "Soar", "Manouvre", "Boat", "Float", "Float_groyne",  "SitStand", "TerLoco",
        "Pecking", "Handling_mussel"
    ]
    
    ind2name = {0: 'Flap', 1: 'ExFlap', 2: 'Soar', 3: 'Boat', 4: 'Float', 5: 'SitStand', 6: 'TerLoco', 7: 'Other', 8: 'Manouvre', 9: 'Pecking', 
    10: 'Looking_food', 11: 'Handling_mussel', 13: 'StandForage', 14: 'xtraShake', 15: 'xtraCall', 16: 'xtra', 17: 'Float_groyne'}

    ignore_labels = {7: 'Other', 10: 'Looking_food', 13: 'StandForage', 14: 'xtraShake', 15: 'xtraCall', 16: 'xtra'}

    merge_labels = {17: 4, 11: 9}

    upper_triangle = np.array([
        [1, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 2, 0, 0, 0, 1, 2, 2, 2],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2],
        [0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])

    # j_data mapping: labels were wrong
    mapping_j = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 8: 10, 9: 11, 10: 13}
    # fmt: on

    rule = upper_triangle + upper_triangle.T - np.diag(upper_triangle.diagonal())
    rule_df = pd.DataFrame(rule, index=labels, columns=labels)

    # fmt: off
    return SimpleNamespace(
        ind2name=ind2name, ignore_labels=ignore_labels, rule_df=rule_df, mapping_j = mapping_j, merge_labels=merge_labels
    )
    # fmt: on


def drop_groups_with_all_neg1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops groups based on the first two columns where every value in the fourth column is -1.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Filtered DataFrame with unwanted groups removed.
    """

    # Use groupby + filter to keep only groups where NOT all values in target_col are -1
    filtered_df = df.groupby([0, 1]).filter(lambda group: not (group[3] == -1).all())
    return filtered_df


def complete_data_from_db(df: pd.DataFrame, df_db: pd.DataFrame) -> pd.DataFrame:
    """
    Complete labeled sensor data by retrieving missing sensor data from df_db

    This function assumes:
    - `df` contains labeled but incomplete sensor data (some sensor indices per device/timestamp).
    - `df_db` contains the full sensor data (all indices), but without labels (dummy label -1 is used).

    For each (device_id, timestamp) pair in `df`, the function:
    - Retrieves all corresponding sensor rows from `df_db`.
    - Fills in known labels from `df`.
    - Assigns label -1 where no label is available.

    Args:
        df (pd.DataFrame): Labeled but incomplete sensor data. Must contain columns [0, 1, 2, 3, ...].
        df_db (pd.DataFrame): Complete sensor data without valid labels. Must contain columns [0, 1, 2, 3, ...].

    Returns:
        pd.DataFrame: Completed DataFrame with all sensor indices for relevant timestamps and device_ids,
                      containing correct labels where available, and -1 where not.
    """
    # Step 1: Get relevant (device_id, timestamp) pairs from df
    device_date_pairs = df[[0, 1]].drop_duplicates()

    # Step 2: Filter df_db to rows matching those pairs
    df_db_filtered = df_db.merge(device_date_pairs, on=[0, 1], how="inner")

    # Step 3: Merge filtered df_db with df to get new labels
    df_labeled = df_db_filtered.merge(
        df[[0, 1, 2, 3, 4, 5, 6, 7]], on=[0, 1, 2], how="left", suffixes=("", "_new")
    )
    # # The merge seems a bit magical and creates own issues with changing the column names and dtypes.
    # # It is faster version of looping over the rows and checking for matches and replacing them.
    # # We need to fix the column names and dtypes afterwards.
    # dt = device_date_pairs.iloc[0]
    # subset_db = df_db[(df_db[0] == dt[0]) & (df_db[1] == dt[1])].copy()
    # subset_df = df[(df[0] == dt[0]) & (df[1] == dt[1])].copy()
    # for i in df[2].values:
    #     ind = subset_db[subset_db[2] == i].index[0]
    #     subset_db.loc[ind, 3] = subset_df[subset_df[2] == i].iloc[0, 3]

    # Step 4: Ensure column names are clean integers again
    # Merge created the string columns "3_new" and "3"
    df_labeled.columns = [
        int(col) if str(col).isdigit() else col for col in df_labeled.columns
    ]

    # Step 5: Overwrite columns with new values if they exist
    # Since there are some rounding differences, I use original columns 4, 5, 6, 7
    # A.combine_first(B): Take values from A, but wherever A has NaN, use values from B instead.
    for col in [3, 4, 5, 6, 7]:
        new_col = f"{col}_new"
        if new_col in df_labeled.columns:
            df_labeled[col] = df_labeled[new_col].combine_first(df_labeled[col])
            df_labeled = df_labeled.drop(columns=new_col)

    df_labeled[3] = df_labeled[3].astype("int64")  # Merge created a float column

    return df_labeled.sort_values([0, 1, 2]).reset_index(drop=True)


def evaluate_and_modify_df(df, rule):
    """
    Applies label evaluation logic on df[3] and potentially modifies labels.
    Returns modified df or None.

    n_uniq_labels are the number of unique labels excluding -1.
    The rules are as follows:
    #### Case 1: `n_uniq_labels > 2`
    - **Reject**

    #### Case 2: `n_uniq_labels == 1`
    - Accept **iff** `|l1| ≥ 5`

    #### Case 3: `n_uniq_labels == 2`
    - If `r == 0`: Reject
    - If `r == 1`: Accept **iff** `|l1| ≥ 5`
    - If `r == 2`: Accept **iff** `|l2| ≥ 5`
    """

    labels = df[3].tolist()
    counts = Counter(labels)
    unique_labels = sorted(counts.keys())
    valid_labels = [label for label in unique_labels if label != -1]
    n_unique = len(valid_labels)

    if n_unique > 2:
        return None

    if n_unique == 0:
        return None  # Only -1s? Reject

    if n_unique == 1:
        l1_label = valid_labels[0]
        if counts[l1_label] >= 5:
            df[3] = l1_label
            return df
        else:
            return None

    if n_unique == 2:
        l1_label = next((val for val in labels if val != -1), None)
        l2_label = [i for i in valid_labels if i != l1_label][0]
        if rule == 0:
            return None
        elif rule == 1:
            if counts[l1_label] >= 5:
                df[3] = l1_label
                return df
            else:
                return None
        elif rule == 2:
            if counts[l2_label] >= 5:
                df[3] = l2_label
                return df
            else:
                return None

    return None


def process_moving_window_given_dt(df, dt, rule_df, ind2name, glen):
    new_df = []
    # df_dt = df[(df[0] == dt[0]) & (df[1] == dt[1]) & (df[3] != -1)].copy()
    df_dt = df[(df[0] == dt[0]) & (df[1] == dt[1])].copy()
    df_dt = df_dt.reset_index(drop=True)
    for start in range(len(df_dt) - glen + 1):
        cut = df_dt.iloc[start : start + glen].copy()
        if len(cut) < glen:
            break

        valid_labels = np.unique(cut[3])
        valid_labels = [i for i in valid_labels if i != -1]

        if len(valid_labels) > 2:
            continue

        if len(valid_labels) == 2:
            l1_label = next((val for val in cut[3] if val != -1), None)
            l2_label = [i for i in valid_labels if i != l1_label][0]
            which_label = rule_df[ind2name[l1_label]][ind2name[l2_label]]
        else:
            which_label = 1

        cut = evaluate_and_modify_df(cut, which_label)
        if cut is not None:
            new_df.append(cut)
    return new_df


def shift_df(df, glen, dts=None):
    rule_df = get_rules().rule_df
    ind2name = get_rules().ind2name

    df = df.sort_values([0, 1]).reset_index(drop=True)

    new_df = []
    if dts is None:
        dts = df[[0, 1]].drop_duplicates().values
    for dt in tqdm(dts):
        new_data = process_moving_window_given_dt(df, dt, rule_df, ind2name, glen)
        new_df.extend(new_data)
    new_df = pd.concat(new_df, ignore_index=True)
    return new_df


def combine_csv_files(csv_files):
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def merge_prefer_valid(*dfs):
    """
    merge and pick the valid (col 3 ≠ -1) row
    """
    df = pd.concat(dfs, ignore_index=True)

    # for each group of (0,1,2), pick the row whose col 3 != -1 if it exists,
    # otherwise pick whichever row is first.
    def pick_valid(sub):
        # boolean mask of “valid” rows
        valid = (sub[3] != -1).to_numpy()
        # argmax gives position of first True, or 0 if none
        return sub.iloc[valid.argmax()]

    return (
        df.groupby([0, 1, 2], as_index=False).apply(pick_valid).reset_index(drop=True)
    )


def group_equal_elements_optimized(df, subset, indices, glen=20):
    hashes = {}
    for idx in tqdm(indices, total=len(indices)):
        crop = df[subset].loc[idx : idx + glen - 1].reset_index(drop=True)
        group_rows = pd.util.hash_pandas_object(crop).values.tobytes()

        if group_rows not in hashes:
            hashes[group_rows] = []
        hashes[group_rows].append(idx)

    # Extract groups of duplicates from the hash map
    groups = [group for group in hashes.values() if len(group) > 1]

    return groups


def drop_duplicates(df, glen=20):
    subset = [0, 1, 2]
    slice = df[subset].iloc[::glen].copy()
    inds = slice[slice.duplicated(keep=False)].index
    if len(inds) == 0:
        print("No duplicates found")
        return df
    print(len(inds))

    groups = group_equal_elements_optimized(df, subset, inds, glen)
    groups = [list(map(int, g)) for g in groups]
    print(groups)

    # Collect all the indices to be dropped
    to_drop = []
    for g in groups:
        if len(g) > 1:
            # Collect indices to drop (ignoring the first one)
            to_drop.extend(range(i, i + glen) for i in g[1:])

    # Flatten the list of indices to drop
    to_drop = [item for sublist in to_drop for item in sublist]

    # Drop all the collected rows at once
    df = df.drop(to_drop)
    return df


def make_data_pipeline(name, input_file, save_path, database_file, change_format):
    """
    pipeline: format, index, map0, mistake, map, drop_neg1, complete, \{combine, shift, drop}

    name: s: set1 (judy), j (json suzzane), m (mat suzzane), w (csv willem)
    """
    # for map_new_labels just use the mapping to -1 for ignore labels

    print(f"Processing {name} data")

    save_path.mkdir(parents=True, exist_ok=True)

    df_db = pd.read_csv(database_file, header=None)

    # Format
    print("Format")
    save_file = save_path / f"{name}_format.csv"
    change_format[name](input_file, save_file)
    df = pd.read_csv(save_file, header=None)

    # Index
    print("Index")
    save_file = save_path / f"{name}_index.csv"
    add_index(df_db, df, save_file)
    df = pd.read_csv(save_file, header=None)

    # Map0
    print("Map0")
    # map0: First mapping due to wrong labels in j_data. The other data remain the same.
    # For consistency, the other mapping is identity.
    if name == "j":
        mapping = get_rules().mapping_j
    else:
        u_label_ids = np.unique(df[3])
        mapping = dict(zip(u_label_ids, u_label_ids))
    save_file = save_path / f"{name}_map0.csv"
    df = map_new_labels(df, mapping, save_file)

    # Misakes
    print("Misakes")
    df = correct_mistakes(df, name)
    save_file = save_path / f"{name}_correct.csv"
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")

    # Map: for ignored labels
    print("Map")
    ignore_labels = get_rules().ignore_labels
    merge_labels = get_rules().merge_labels
    u_label_ids = np.unique(df[3])
    mapping = {idx: -1 if idx in ignore_labels else idx for idx in u_label_ids}
    mapping.update(merge_labels)
    save_file = save_path / f"{name}_map.csv"
    df = map_new_labels(df, mapping, save_file)

    # Drop groups with all label -1 (invalid)
    print("Drop groups with all -1( invalid)")
    df = drop_groups_with_all_neg1(df)

    # Complete
    print("Complete")
    df = complete_data_from_db(df, df_db)
    save_file = save_path / f"{name}_complete.csv"
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")

    print("Done")


def make_combined_data_pipeline(input_path: Path, save_path: Path, filenames: list):
    """
    pipeline: \{format, index, map0, mistake, map, drop_neg1, complete}, combine, shift, drop
    """
    print("Combined")
    dfs = [pd.read_csv(input_path / i, header=None) for i in filenames]
    save_file = save_path / "combined.csv"
    df = merge_prefer_valid(*dfs)
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")
    del dfs

    # print("Shift")
    df = shift_df(df, 20)
    save_file = save_path / "shift.csv"
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")
    # sorted({k: v//20 for k, v in Counter(df[3].values).items()}.items())

    print("Drop duplicates")
    save_file = save_path / "drop.csv"
    df = drop_duplicates(df)
    df.to_csv(save_file, index=False, header=None, float_format="%.6f")
    print("Done")


"""
# Old to be removed
# dpath = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")
# json_file = Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne/combined.json")
# save_file = Path("/home/fatemeh/Downloads/bird/data/final/proc/j_data_format.csv")
# new2old_labels = {5: 0, 4: 1, 3: 2, 2: 4, 1: 5, 0: 6, 7: 7, 6: 8, 9: 9, 10: 9}
# ignored_labels = [8, 14, 15, 16, 17]
# # bd.combine_jsons_to_one_json(list(dpath.glob("*json")), json_file)
"""

# dfs = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/s_map.csv", header=None)
# dfj = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/j_map.csv", header=None)
# dfm = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/m_map.csv", header=None)
# dfw = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/w_map.csv", header=None)
# dts = dfs[[0,1]].drop_duplicates().reset_index(drop=True)
# dtw = dfw[[0,1]].drop_duplicates().reset_index(drop=True)
# dtj = dfj[[0,1]].drop_duplicates().reset_index(drop=True)
# msw = pd.merge(dts, dtw).reset_index(drop=True)
# msj = pd.merge(dts, dtj).reset_index(drop=True)
# mwj = pd.merge(dtw, dtj).reset_index(drop=True)
# mswj = pd.merge(msw, msj).reset_index(drop=True)
# dfswjm = bdp.merge_prefer_valid(dfs, dfj, dfm, dfw)
# >>> len(dts), len(dtj), len(dtm), len(dtw)
# (1526, 1854, 706, 1426)
# >>> len(dfs), len(dfj), len(dfm), len(dfw)
# (71842, 90668, 32790, 67807)
# >>> len(dfswjm)
# 105692

# change_format = {"s": change_format_json_file, "j": change_format_json_file, "m": change_format_mat_files, "w": change_format_csv_files}
# save_path = Path("/home/fatemeh/Downloads/bird/data/final/proc2")
# database_file = Path("/home/fatemeh/Downloads/bird/data/final/orig/all_database_final.csv")
# name_input_files = [
# ("s", Path("/home/fatemeh/Downloads/bird/data/set1/data/combined.json")),
# ("j", Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne/combined.json")),
# ("m", Path("/home/fatemeh/Downloads/bird/data/data_from_Susanne")),
# ("w", Path("/home/fatemeh/Downloads/bird/data/data_from_Willem"))
# ]
# [make_data_pipeline(name, input_file, save_path, database_file, change_format) for name, input_file in name_input_files]
# filenames = [f"{i}_complete.csv" for i in ["s", "j", "m", "w"]]
# make_combined_data_pipeline(save_path, save_path, filenames)
# print("Done")


def get_s_j_w_m_data_from_database(data, save_file, database_url, glen=20):
    """
    Get all the data from the database (1930 requests)
    """
    # data = pd.concat((df_s, df_j, df_w, df_m), axis=0, ignore_index=True)
    unique_dt = (
        data[[0, 1]].drop_duplicates().sort_values(by=[0, 1]).reset_index(drop=True)
    )

    file = open(save_file, "w")
    for _, row in tqdm(unique_dt.iterrows(), total=len(unique_dt)):
        device_id, start_time = list(row)
        try:
            igs, idts, _ = bd.get_data(
                database_url, device_id, start_time, start_time, glen
            )
        except Exception as e:
            print(f"Error during data processing or saving results: {e}")
            continue
        if len(igs) == 0:
            print("Not in database", device_id, start_time)
            continue

        indices = idts[:, 0]
        sel_igs = np.round(igs, 6)
        for i, index in zip(sel_igs, indices):
            item = (
                f"{device_id},{start_time},{index},-1,{i[0]:.6f},{i[1]:.6f},"
                f"{i[2]:.6f},{i[3]:.6f}\n"
            )
            file.write(item)
        file.flush()
    file.close()


def map_to_nearest_divisible_20(start, end):
    # The numpy and python round function, map the border value such as .5, 1.5, 2.5, 3.5
    # and so on to the nearest even number 0., 2., 2., 4..
    # So border values such as 10, 30, 50, 70, 90 map differently 0, 40, 40, 80, 80, using below code:
    # return [int(round(i / 20) * 20) for i in [start, end]]
    # This is fixed by `i = i + .001`
    new_values = []
    for i in [start, end]:
        if i % 20 == 10:
            i = i + 0.001
        new_values.append(int(round(i / 20) * 20))
    return new_values


def check_batches(df: pd.DataFrame, batch_size=20):
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i + batch_size]
        assert len(batch) == batch_size

        assert batch[0].nunique() == 1, "nonunique device id"
        assert batch[1].nunique() == 1, "nonunique time"
        assert batch[3].nunique() == 1, "nonunique label"
        assert batch[7].nunique() == 1, "nonunique GPS"

        assert all(np.diff(batch[2].values) == 1), "not consecutive index"


def get_start_end_inds(df, dt):
    """
    Indices are data indices (df[3]).
    """
    cut = df[(df[0] == dt[0]) & (df[1] == dt[1])].copy()
    u_labels = np.unique(cut[3])
    start_end_inds = dict()
    for u_label in u_labels:
        sel_inds = cut[cut[3] == u_label][2].values
        start_end_inds[sel_inds[0], sel_inds[-1]] = u_label
    start_end_inds = dict(sorted(start_end_inds.items()))
    return start_end_inds
