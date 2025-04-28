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
