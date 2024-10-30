# Cheet sheet pandas

### Simple example on grouping

- group: `filter | apply | .transform('size')`
- specific on group:  `.get_group | .groups | .groups.keys()`
- general on datafram or group: `.sort_index() | .reset_index(drop=True), .reset_index(name="something")` 
    `|(..., ignore_index=True) | .copy() | sort_values`
    `| .round(4) | .drop(index) | pd.DataFrame | pd.read_csv | .to_csv | .drop_duplicates() | .duplicated()`

```python
import pandas as pd

data = {
    "A": [1, 1, 2, 2, 3, 3, 3],
    "B": [10, 10, 20, 20, 30, 30, 30],
    "C": [5, 6, 7, 8, 9, 10, 11],
}
df = pd.DataFrame(data)
df = pd.read_csv(csv_file, header=None)
df.to_csv(save_file, index=False, header=None, float_format="%.6f")

df.drop_duplicated()
# .sort_values(by=[0,1], ignore_index=True)
# .sort_values(by=[0,1]).reset_index(drop=True)

grouped = df.groupby("A")
for name, group in grouped:
    pass
grouped.groups.keys() # .groups is a dictionary
grouped.size()
group = grouped.get_group((2,)) # get a specific group by a name
group.sort_values(by=["A","C"], ascending=[True, True])

group_sizes = grouped.size()
groups_with_size_2 = group_sizes[group_sizes == 2]
df[df.groupby(["A", "B"]).transform('size') > 2]
groupby(["A", "B"]).size().reset_index(name='s')

filtered = df.groupby("A").filter(lambda x: len(x) > 2)

def modify_column(group):
    group["C"] = group["C"] * 2
    return group


# with warning
modified_df = df.groupby("A").apply(modify_column).reset_index(drop=True)
# solution 1: no warning
modified_df1 = (
    df.groupby("A").apply(modify_column, include_groups=False).reset_index(drop=True)
)
modified_df1 = pd.concat((df["A"], modified_df1), axis=1)
# solution 2: no warning
modified_df2 = (
    df.groupby("A")[["A", "B", "C"]].apply(modify_column).reset_index(drop=True)
)

modified_df.equals(modified_df2)  # also for other ones
"""
filtered without reset_index
   A   B   C
4  3  30   9
5  3  30  10
6  3  30  11
modified_df without reset_index
A             
1 0  1  10  10
  1  1  10  12
2 2  2  20  14
  3  2  20  16
3 4  3  30  18
  5  3  30  20
  6  3  30  22
"""
```

### Grouping on real data

<summary>[Click to expand]</summary>
<details>

```python
import pandas as pd

df = pd.read_csv("/home/fatemeh/Downloads/s_data.csv", header=None)

grouped = df.groupby([0, 1, 2, 3], sort=False)  # otherwise group is sorting
invalid_groups = grouped.filter(lambda x: len(x) != 20)
invalid_indices = invalid_groups.index.tolist()
invalid_rows = df.loc[invalid_indices]


# Function to modify the third column for each group
def modify_index(group):
    if len(group) == 20:
        start_value = int(
            group.iloc[0, 2]
        )  # Get the starting value from the third column (df[2])
        group[2] = range(
            start_value, start_value + 20
        )  # Replace with sequential numbers
    return group


modified_df = grouped.apply(modify_index).reset_index(drop=True)  # with warning
"""
# solution 1: no warning > doesn't work
# When use include_groups=False, pandas passes only the non-grouping columns (in this case, 4, 5, 6, 7) 
# to the function. This means group[2] no longer refers to the third column, which causes the modify_index 
# function to operate on the wrong column.
modified_df1 = grouped.apply(modify_index, include_groups=False).reset_index(drop=True)
modified_df1 = pd.concat((df[[0,1]], modified_df1[2], df[[3,4,5,6,7]]),axis=1)
"""
# solution 2: no warning
modified_df2 = (
    grouped[[0, 1, 2, 3, 4, 5, 6, 7]].apply(modify_index).reset_index(drop=True)
)

df[[0, 1, 3, 4, 5, 6, 7]].equals(
    modified_df[[0, 1, 3, 4, 5, 6, 7]]
)  # also for other ones

modified_df2.to_csv(
    "/home/fatemeh/Downloads/s_data_modified.csv",
    index=False,
    header=None,
    float_format="%.8f",
)
```
</details>
