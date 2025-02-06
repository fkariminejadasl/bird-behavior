# Cheet Sheet Pandas

### List of Operations

- general on dataframe or group: 
    - `pd.DataFrame | pd.read_csv | .to_csv`
    - `.iloc, .loc| .iat, .at | .drop`
    - `.index | .sort_index() | .drop(index) | .reset_index(drop=True), .reset_index(name="something") |.some_operation(..., ignore_index=True) | .rename`
    - `len(df), df.shape, df.size | .equals | pd.concat | .round(4) | .copy()`
    - `sort_values`
    - `.drop_duplicates() | .duplicated()`
- group: 
    - `.get_group | .groups | .groups.keys()`
    - `apply | filter | .transform('size')`

### Read and Write
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
```

### General Operations


#### Get elements, subset/slicing, removing

```python
# Subsets: confusing
# This is confusing, since if columns as specifically asked, it will be column-based subset. 
# But when some condition applies it becomes index based.
df[["A", "B"]], df2[[0,1]] # by columns
df["A"], df2[0] # by single column
df[df["A"]>2], df[df[0]>2] # by index
df[[False,  True, False, False,  True, False]] # by index

# Slicing
df.iloc[] # by index. Similar to numpy
df.loc[] # by label


# Single value
df.iat[]
df.at[]

# Remove rows
df.drop([0, 1]))

# Remove columns
df.drop(columns=['B'])
```

#### Index

After removing some rows, pandas keeps the original index. Reset the index so that it starts from 0 and increases sequentially. For example, remove some rows and reset the indices as shown below:

```python
a = pd.concat([df[df[3] == i].iloc[:len_data] for i in keep_labels])
a = a.reset_index(drop=True)
```

```bash
# values after removing some rows
          0                    1   2  3         4         5         6         7
240     533  2012-05-15 03:29:24   0  0 -0.358179 -0.445566  1.804016  9.817386
241     533  2012-05-15 03:29:24   1  0  0.192266 -0.319335  1.707204  9.817386

# values after reset the index
          0                    1   2  3         4         5         6         7
0       533  2012-05-15 03:29:24   0  0 -0.358179 -0.445566  1.804016  9.817386
1       533  2012-05-15 03:29:24   1  0  0.192266 -0.319335  1.707204  9.817386
```

```python
# Get index
.index

# Reset index
.some_operation(..., ignore_index=True)
.reset_index(name='s')
.reset_index(drop=True)

# Rename column
.rename(columns={"A":"A2"})
```

#### Basic operations

```python
# Create numpy object
df.values # The same as np.array(df)

len(df), df.shape, df.size # 7, (7,3), 21

df1.equals(df2)
pd.concat()
df.copy()
# Float with precision 2 (e.g .45)
df.round(2)
```

#### Sort

Sort by columns 0 and then 1:
```python
df2 = pd.DataFrame({0:[2,3,1,2,3,1],1:["2014-04-20 12:58:41","2012-05-15 03:10:11","2014-05-20 13:03:30","2014-05-20 12:58:41","2012-06-15 03:10:11","2014-06-20 13:03:30"], 2:[0,1,1,1,0,0]})
df2.sort_values(by=[0,1], ignore_index=True)
df2.sort_values(by=[0,1]).reset_index(drop=True)
```

#### Duplicates

``` python
df.drop_duplicated()
```

### Group Operations

```python
grouped = df.groupby("A")

# Print elements
for name, group in grouped:
    pass

# Get elements
grouped.groups.keys() # .groups is a dictionary
group = grouped.get_group((2,)) # get a specific group by a name

# Sort group
group.sort_values(by=["A","C"], ascending=[True, True])

# Group sizes
group_sizes = grouped.size()
groups_with_size_2 = group_sizes[group_sizes == 2]
groupby(["A", "B"]).size().reset_index(name='s')

# Group transform, filter, modify
df[df.groupby(["A", "B"]).transform('size') > 2]
filtered = df.groupby("A").filter(lambda x: len(x) > 2)

def modify_column(group):
    group["C"] = group["C"] * 2
    return group

df.groupby("A")[["A", "B", "C"]].apply(modify_column).reset_index(drop=True)
```

#### Simple example on grouping

The data is:

```bash
          0                    1   2  3         4         5         6         7
0       533  2012-05-15 03:29:24   0  0 -0.358179 -0.445566  1.804016  9.817386
1       533  2012-05-15 03:29:24   1  0  0.192266 -0.319335  1.707204  9.817386
```

We group every 20 items. So first we add extra column of indexes, which the data become:

```bash
          0                    1   2  3         4         5         6         7     8
0       533  2012-05-15 03:29:24   0  0 -0.358179 -0.445566  1.804016  9.817386     0
1       533  2012-05-15 03:29:24   1  0  0.192266 -0.319335  1.707204  9.817386     0
```

Now we group based on the new column. We can also group by column 0 and 1 since they are the same values. Then we assert that the 3rd column is unique.

``` python
a[8] = a.index //20 # add extra column
grouped = a.groupby([0,1,8]) # group by columns 0, 1, 8
for n, g in grouped:
    assert len(np.unique(g[3]))==1 # assert unique value
```

#### Simple example on grouping

```python
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
```

```bash
data
   A   B   C
0  1  10   5
1  1  10   6
2  2  20   7
3  2  20   8
4  3  30   9
5  3  30  10
6  3  30  11
filtered without reset_index
   A   B   C
4  3  30   9
5  3  30  10
6  3  30  11
filtered with reset_index
   A   B   C
0  3  30  18
1  3  30  20
2  3  30  22
modified_df without reset_index
     A   B   C
A             
1 0  1  10  20
  1  1  10  24
2 2  2  20  28
  3  2  20  32
3 4  3  30  36
  5  3  30  40
  6  3  30  44
modified_df with reset_index
   A   B   C
0  1  10  20
1  1  10  24
2  2  20  28
3  2  20  32
4  3  30  36
5  3  30  40
6  3  30  44
```

#### Grouping on real data

<summary>[Click to expand]</summary>
<details>

Modify the indices (df[2]) to follow an increasing order instead of a fixed number. Currently, all groups have the same starting index (e.g., 20). After this change, the indices will follow an increasing order.

``` python
def modify_index(group):
    if len(group) == 20:
        # Get the starting value from the third column (df[2])
        start_value = int(group.iloc[0, 2])
        # Replace with sequential numbers
        group[2] = range(start_value, start_value + 20)
    return group


df = pd.read_csv("s_data.csv", header=None)
grouped = df.groupby([0, 1, 2, 3], sort=False)  # otherwise group is sorting
modified_df2 = (
    grouped[[0, 1, 2, 3, 4, 5, 6, 7]].apply(modify_index).reset_index(drop=True)
)
modified_df2.to_csv("s_data_modified.csv", index=False, header=None, float_format="%.6f")
```

``` python
>>> df
         0                    1   2  3         4         5         6         7
0      533  2012-05-15 03:10:11  20  6 -0.327772 -0.255841  0.807911  0.020033
1      533  2012-05-15 03:10:11  20  6 -0.469671 -0.109957  0.856317  0.020033
>>> modified_df2
         0                    1   2  3         4         5         6         7
0      533  2012-05-15 03:10:11  20  6 -0.327772 -0.255841  0.807911  0.020033
1      533  2012-05-15 03:10:11  21  6 -0.469671 -0.109957  0.856317  0.020033

"""
# solution 1: no warning > doesn't work
# When use include_groups=False, pandas passes only the non-grouping columns (in this case, 4, 5, 6, 7) 
# to the function. This means group[2] no longer refers to the third column, which causes the modify_index 
# function to operate on the wrong column.
modified_df1 = grouped.apply(modify_index, include_groups=False).reset_index(drop=True)
modified_df1 = pd.concat((df[[0,1]], modified_df1[2], df[[3,4,5,6,7]]),axis=1)
"""

# Identify invalid groups
invalid_groups = grouped.filter(lambda x: len(x) != 20)
invalid_indices = invalid_groups.index.tolist()
invalid_rows = df.loc[invalid_indices]
```
</details>


### Reference

- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)