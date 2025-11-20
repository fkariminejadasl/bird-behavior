I have 186 parquet files. Each file has different sizes. The largest file has two row groups one with about 300MB data and the other 100MB, each has 1048576, 462367 lines. Each line is 80 np.float32 array. The example of the other file is only one row group with 9296 lines, which is about 1MB data. 

Each parquet file created from multiple CSV files roughly the same size (30MB). Each csv file contains device id, date, index, label, imu_x, imu_y, imu_z and GPS 2d speed. Per 20 lines, all the values are the same except imu_{x,y,z}, gps 2d speed and indices. For training, we only need imu_{x,y,z}, gps 2d speed. So, in parquet format, every 20 rows share the same device and date are presented as one line of {imu, gps} x 20, so 80 np.float32. All the csv files of the same device are combined into one parquet file. So, per device, there is one parquet file and multiple CSV file.

For example, device id 298 has several csv files with names like 298_0.csv, 298_1.csv, ...,298_n.csv​. These files are roughly the same size but different number of lines. They are combined to one single parquet file (e.g. 298.parquet ) and took every 20 lines and took column 4-7 and flatten them into 20x4=80 np.float32 numpy array. Here is the code:


```
def curate_data(df: pd.DataFrame):
    """
    Curate the input dataframe by performing below preprocessing steps.
    - GPS 2D speed smaller than 30 m/s
    - Clip IMU x, y, z values between -2, 2
    - Normalize GPS speed by gps_scale

    Parameters:
    ===========
    df : pd.DataFrame
        Input dataframe containing raw data.
    Return:
    =======
    pd.DataFrame
    """
    # GPS 2D speed smaller than 30 m/s
    df = df[df[7] < 30.0].copy()

    # Clip IMU x, y, z values between -2, 2
    df[[4, 5, 6]] = df[[4, 5, 6]].clip(-2.0, 2.0)

    # Normalize GPS speed by gps_scale
    gps_scale = 22.3012351755624
    df[7] = df[7] / gps_scale

    return df


def write_only_gimu_float32_norm_gps_batch(csv_files, parquet_file):
    """
    save only gimu (4,5,6,7) in float32 and normalize gps speed
    """
    count = 0
    arr = pa.array([], type=pa.list_(pa.float32()))
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file, header=None)

        # Curate data
        df = curate_data(df)

        # Extract only gimu columns
        gimus = df[[4, 5, 6, 7]].to_numpy(dtype=np.float32)
        # Reshape into (num_chunks, 20 * 4)
        n_full_chunks = gimus.shape[0] // 20
        gimus_flat = gimus.reshape(n_full_chunks, 20 * 4)

        # Convert to Arrow array of float32 lists
        new_arr = pa.array(gimus_flat.tolist(), type=pa.list_(pa.float32()))
        arr = pa.concat_arrays((arr, new_arr))
        print(len(new_arr))
        count += len(new_arr)
    print(len(arr), count)
    table = pa.Table.from_arrays([arr], names=["gimu"])
    # df.to_parquet generates larger file compare to pq.write_table
    pq.write_table(table, parquet_file)

if "__main__" == __name__:
    parquet_path = Path("/home/fatemeh/Downloads/bird/data/ssl/ssl20parquet")
    parquet_path.mkdir(parents=True, exist_ok=True)

    csv_path = Path("/home/fatemeh/Downloads/bird/data/ssl/ssl20")
    devices = np.unique([int(p.stem.split("_")[0]) for p in csv_path.glob("*.csv")])

    for device in tqdm(devices):
        csv_files = csv_path.glob(f"{device}_*.csv")
        csv_files = sorted(csv_files, key=lambda x: int(x.stem.split("_")[1]))
        parquet_file = parquet_path / f"{device}.parquet"
        write_only_gimu_float32_norm_gps_batch(csv_files, parquet_file)
```

Currently we read the data in memory. There are roughtly 25 million data points, which is roughly 7.6GB data (25e6 (data points) x 80 (each line of data) x 4 (np.float32)). Here is the code to read the data in the memory.


```
gimus = []
parquet_files = cfg.data_path.glob("*.parquet")
for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    data = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 20, 4)))
    print(parquet_file.stem, data.shape)
    gimus.append(data)
gimus = np.vstack(gimus)
del df, data
gc.collect()
gimus = np.ascontiguousarray(gimus)

class BirdDataset(Dataset):
    def __init__(
        self,
        all_measurements: np.ndarray,  # NxLxC
        ldts: np.ndarray = None,  # Nx3
        transform=None,
        channel_first=True,
    ):
        """
        dtype: all_measurements np.float32
        dtype: ldts np.int64 or None (if no labels are provided)
        :param channel_first: If True, data is returned in CxL format (channel-first). Otherwise, LxC (channel-last).
        """
        # data: NxLxC C=4
        self.data = np.ascontiguousarray(all_measurements, dtype=np.float32)
        self.has_label = ldts is not None  # Check if labels are provided
        if self.has_label:
            self.ldts = np.ascontiguousarray(ldts, dtype=np.int64) # Nx3

        self.transform = transform
        self.channel_first = channel_first  # Flag for channel arrangement

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        data = self.data[ind]  # LxC
        data = torch.from_numpy(data)  # torch
        if self.transform:
            data = self.transform(data)

        # Rearrange channels if channel_first is True
        if self.channel_first:
            data = data.transpose(1, 0)  # LxC -> CxL
        if self.has_label:
            ldt = torch.from_numpy(self.ldts[ind])  # 3 torch
            return data, ldt  # Return both data and label

dataset = BirdDataset(gimus, channel_first=False)

train_loader = DataLoader(
    dataset,
    batch_size=min(cfg.batch_size, len(train_dataset)),
    shuffle=True,
    num_workers=cfg.num_workers,
    drop_last=False,
    pin_memory=True,  # fast but more memory
)
```


We want to change my BirdDataset to be able to manage to have data not in the memory.


We have few solutions:

Solutions 1: Memory-efficient dataset that loads data on-demand from parquet files

```
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import bisect
from typing import Optional, List, Tuple
import pyarrow.parquet as pq
from pathlib import Path
from torch.utils.data import DataLoader


# Solution: Memory-efficient dataset that loads data on-demand from parquet files
class MemoryEfficientBirdDataset(Dataset):
    def __init__(
        self,
        parquet_files: List[Path],
        ldts: np.ndarray = None,
        transform=None,
        channel_first=True,
        cache_size: int = 5,  # Number of files to keep in memory
        use_pyarrow: bool = True,  # PyArrow is faster than pandas for single-row access
    ):
        """
        Memory-efficient dataset that loads data on-demand from parquet files.
        
        Args:
            parquet_files: List of parquet file paths
            ldts: Labels array (Nx3) or None
            transform: Optional transform function
            channel_first: If True, data is returned in CxL format
            cache_size: Number of parquet files to keep in memory (per worker)
            use_pyarrow: Use PyArrow for faster single-row access
        """
        self.parquet_files = sorted(parquet_files)
        self.transform = transform
        self.channel_first = channel_first
        self.cache_size = cache_size
        self.use_pyarrow = use_pyarrow
        
        # Build index: map global index to (file_idx, row_in_file)
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        print("Building index from parquet files...")
        for pf in self.parquet_files:
            if use_pyarrow:
                pf_file = pq.ParquetFile(pf)
                length = pf_file.metadata.num_rows
            else:
                df = pd.read_parquet(pf)
                length = len(df)
            self.file_lengths.append(length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        
        self.total_length = self.cumulative_lengths[-1]
        print(f"Total samples: {self.total_length} from {len(self.parquet_files)} files")
        
        # Labels
        self.has_label = ldts is not None
        if self.has_label:
            assert len(ldts) == self.total_length, \
                f"Label length {len(ldts)} doesn't match data length {self.total_length}"
            self.ldts = np.ascontiguousarray(ldts, dtype=np.int64)
        
        # Cache for loaded data (per worker process)
        self._cache = {}
        self._cache_order = []  # LRU tracking
    
    def __len__(self):
        return self.total_length
    
    def _find_file_and_row(self, idx: int) -> Tuple[int, int]:
        """Find which file and row within that file corresponds to global index."""
        # Binary search in cumulative lengths
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        row_in_file = idx - self.cumulative_lengths[file_idx]
        return file_idx, row_in_file
    
    def _load_file(self, file_idx: int):
        """Load a parquet file into cache."""
        if file_idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._cache[file_idx]
        
        # Load new file
        parquet_file = self.parquet_files[file_idx]
        
        if self.use_pyarrow:
            # PyArrow is faster for random access
            pf = pq.ParquetFile(parquet_file)
            table = pf.read()
            # Convert to numpy for faster indexing
            data = np.vstack([np.array(x).reshape(-1, 20, 4) for x in table['gimu'].to_pylist()])
        else:
            df = pd.read_parquet(parquet_file)
            data = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 20, 4)))
        
        data = np.ascontiguousarray(data, dtype=np.float32)
        
        # Add to cache
        self._cache[file_idx] = data
        self._cache_order.append(file_idx)
        
        # Evict oldest if cache is full
        if len(self._cache) > self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
        
        return data
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")
        
        # Find which file and row
        file_idx, row_in_file = self._find_file_and_row(idx)
        
        # Load file (from cache if available)
        file_data = self._load_file(file_idx)
        
        # Get the specific row
        data = file_data[row_in_file]  # Shape: (20, 4) or (L, C)
        data = torch.from_numpy(data)
        
        if self.transform:
            data = self.transform(data)
        
        # Rearrange channels if needed
        if self.channel_first:
            data = data.transpose(1, 0)  # LxC -> CxL
        
        if self.has_label:
            ldt = torch.from_numpy(self.ldts[idx])
            return data, ldt
        
        return data


# Example usage:
if __name__ == "__main__":
    # Get parquet files
    data_path = Path("/home/fatemeh/Downloads/bird/data/ssl/parquetmini/")
    parquet_files = list(data_path.glob("*.parquet"))
    
    # Create dataset
    dataset = MemoryEfficientBirdDataset(
        parquet_files=parquet_files,
        ldts=None,  # or your labels array
        channel_first=False,
        cache_size=5,  # Keep 5 files in memory per worker
        use_pyarrow=True,
    )
    
    x0 = dataset[0]
    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Each worker has its own cache
        drop_last=False,
        pin_memory=True,
    )
    
    # Training loop
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, tuple):
            inputs, labels = data
        else:
            inputs = data
        print(f"Batch {batch_idx}: {inputs.shape}")
```

Solution 2: Single Memory map

```
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class BirdMemmapDataset(Dataset):
    def __init__(
        self,
        memmap_path: str,
        num_samples: int,
        ldts: np.ndarray = None,
        transform=None,
        channel_first: bool = True,
        L: int = 20,
        C: int = 4,
    ):
        # Open memory mapped file
        self.data = np.memmap(
            memmap_path,
            mode="r",
            dtype="float32",
            shape=(num_samples, L, C),
        )

        self.has_label = ldts is not None
        if self.has_label:
            self.ldts = np.ascontiguousarray(ldts, dtype=np.int64)

        self.transform = transform
        self.channel_first = channel_first

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ind):
        # This slices into memmap, not RAM
        data = self.data[ind]  # L x C
        data = torch.from_numpy(data)

        if self.transform:
            data = self.transform(data)

        if self.channel_first:
            data = data.transpose(1, 0)  # L x C -> C x L

        if self.has_label:
            ldt = torch.from_numpy(self.ldts[ind])
            return data, ldt

        return data


data_path = Path("/home/fatemeh/Downloads/bird/data/ssl/parquetmini/")
parquet_files = sorted(data_path.glob("*.parquet"))


# 1. First pass: compute total number of rows (samples)
file_row_counts = []
for pf_path in parquet_files:
    pf = pq.ParquetFile(pf_path)
    n_rows = pf.metadata.num_rows
    file_row_counts.append(n_rows)

total_rows = sum(file_row_counts)
print("Total samples:", total_rows)

L, C = 20, 4  # from your description: 80 values = 20 x 4

# 2. Create memmap file
memmap_path = "/home/fatemeh/Downloads/bird/data/ssl/parquetmini/gimus_memmap.dat"
gimus_mm = np.memmap(memmap_path, mode="w+", dtype="float32",
                     shape=(total_rows, L, C))

# 3. Fill it sequentially
offset = 0
for pf_path, n_rows in zip(parquet_files, file_row_counts):
    print("Processing", pf_path)
    pf = pq.ParquetFile(pf_path)
    table = pf.read(columns=["gimu"])
    # table["gimu"] is an array of your 80-length arrays
    arrs = [np.array(x).reshape(-1, L, C) for x in table["gimu"].to_pylist()]
    data = np.vstack(arrs)
    assert data.shape[0] == n_rows

    gimus_mm[offset:offset + n_rows] = data
    offset += n_rows

# Flush to disk
del gimus_mm
print("Finished creating memmap:", memmap_path)


num_samples = 16431 # total_rows  # from preprocessing step
dataset = BirdMemmapDataset(
    memmap_path="/home/fatemeh/Downloads/bird/data/ssl/parquetmini/gimus_memmap.dat",
    num_samples=num_samples,
    ldts=None,
    channel_first=False,
)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)

# Training loop
for batch_idx, data in enumerate(train_loader):
    if isinstance(data, tuple):
        inputs, labels = data
    else:
        inputs = data
    print(f"Batch {batch_idx}: {inputs.shape}")
```

Solution 3: Single Memory map

```
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import bisect
from typing import Optional, List, Tuple
import pyarrow.parquet as pq
from pathlib import Path
from torch.utils.data import DataLoader


def convert_parquets_to_memmap(parquet_files, output_dir):
    """
    Convert parquet files to memory-mapped numpy arrays for ultra-fast access.
    This trades disk space for speed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'files': [],
        'total_length': 0,
    }
    
    for pf in tqdm(parquet_files, desc="Converting to memmap"):
        df = pd.read_parquet(pf)
        data = np.vstack(df["gimu"].apply(lambda x: x.reshape(-1, 20, 4)))
        data = np.ascontiguousarray(data, dtype=np.float32)
        
        # Save as memory-mapped file
        mmap_file = output_dir / f"{pf.stem}.npy"
        np.save(mmap_file, data)
        
        metadata['files'].append({
            'original': str(pf),
            'mmap': str(mmap_file),
            'length': len(data),
            'shape': data.shape,
        })
        metadata['total_length'] += len(data)
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Converted {len(parquet_files)} files to {output_dir}")
    print(f"Total samples: {metadata['total_length']}")


class MemmapBirdDataset(torch.utils.data.Dataset):
    """
    Ultra-fast dataset using memory-mapped numpy arrays.
    No caching needed - OS handles memory mapping efficiently.
    """
    def __init__(
        self,
        mmap_dir: Path,
        ldts: np.ndarray = None,
        transform=None,
        channel_first=True,
    ):
        mmap_dir = Path(mmap_dir)
        
        # Load metadata
        with open(mmap_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Open all memmap files (doesn't load into memory)
        self.mmaps = []
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        for file_info in metadata['files']:
            mmap_file = Path(file_info['mmap'])
            # Open in read-only mode - OS handles caching
            mmap_array = np.load(mmap_file, mmap_mode='r')
            self.mmaps.append(mmap_array)
            self.file_lengths.append(file_info['length'])
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + file_info['length']
            )
        
        self.total_length = self.cumulative_lengths[-1]
        self.transform = transform
        self.channel_first = channel_first
        
        # Labels
        self.has_label = ldts is not None
        if self.has_label:
            assert len(ldts) == self.total_length
            self.ldts = np.ascontiguousarray(ldts, dtype=np.int64)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Binary search to find file
        import bisect
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        row_in_file = idx - self.cumulative_lengths[file_idx]
        
        # Access memmap (extremely fast - OS handles caching)
        data = self.mmaps[file_idx][row_in_file].copy()  # Copy to make it writable
        data = torch.from_numpy(data)
        
        if self.transform:
            data = self.transform(data)
        
        if self.channel_first:
            data = data.transpose(1, 0)
        
        if self.has_label:
            ldt = torch.from_numpy(self.ldts[idx])
            return data, ldt
        
        return data


# Example: Convert and use
if __name__ == "__main__":
    # Step 1: One-time conversion
    parquet_path = Path("/home/fatemeh/Downloads/bird/data/ssl/parquetmini/")
    mmap_path = Path("/home/fatemeh/Downloads/bird/data/ssl/parquetmini/mmpap")
    parquet_files = list(parquet_path.glob("*.parquet"))
    
    convert_parquets_to_memmap(parquet_files, mmap_path)
    
    # Step 2: Use memmap dataset (much faster)
    dataset = MemmapBirdDataset(
        mmap_dir=mmap_path,
        ldts=None,
        channel_first=False,
    )

    x0 = dataset[0]
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Training loop
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, tuple):
            inputs, labels = data
        else:
            inputs = data
        print(f"Batch {batch_idx}: {inputs.shape}")
```


Solution 4: randomize the data and make own dataloader. Exxample in in nanochat using fineweb-edu, where each file (shard) is 100 MB.


From solusion 1, 2, 3, the solution 2 is the best. UThe data can be pre-shuffled or custom sample such as below beused:

```
import numpy as np
from torch.utils.data import Sampler

class BlockRandomSampler(Sampler):
    def __init__(self, num_samples, block_size, generator=None):
        self.num_samples = num_samples
        self.block_size = block_size
        self.generator = generator

    def __iter__(self):
        n_blocks = (self.num_samples + self.block_size - 1) // self.block_size
        blocks = np.arange(n_blocks)
        rng = np.random.default_rng() if self.generator is None else self.generator
        rng.shuffle(blocks)

        for b in blocks:
            start = b * self.block_size
            end = min(start + self.block_size, self.num_samples)
            for i in range(start, end):
                yield int(i)

    def __len__(self):
        return self.num_samples

sampler = BlockRandomSampler(num_samples=len(dataset), block_size=4096)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=BlockRandomSampler(len(dataset), block_size), # for pre-shuffle, don't use
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,  # recommended
```

It seems to me the solution 1, 2, 3 are not efficient. Solution 2 creates one huge mmap which is not practical, and solution 3 with per device mmap also requires many times opening of the same file because of the dataloader __getitem__, which is similar problem just using solution 1 with parqet format, though solution 3 is better than solution 1. Solution 4 seems the best way and then solution 2. 

