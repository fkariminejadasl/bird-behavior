import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomJitter(nn.Module):
    """Add element-wise Gaussian noise N(0, σ²) to the first 3 dims;
    to the 4th dim, apply one single random offset (or zero if you prefer)."""

    def __init__(self, sigma: float = 0.03):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        noise = torch.randn_like(x) * self.sigma
        if x.size(1) == 4:
            # draw one scalar noise for the last column
            last_noise = torch.randn((), device=x.device, dtype=x.dtype) * self.sigma
            noise[:, 3] = last_noise
        return x + noise


class RandomScaling(nn.Module):
    """Scale each feature channel by a factor ~ N(1, σ²)."""

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        scales = (
            torch.randn(x.size(1), device=x.device, dtype=x.dtype) * self.sigma + 1.0
        )
        return x * scales.unsqueeze(0)


# Python implementation
# =====================

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import resample


def jitter(x, sigma=0.03):
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1],))
    return x * factor


def time_warp(x, sigma=0.2):
    orig_steps = np.arange(x.shape[0])
    random_warp = np.random.normal(loc=1.0, scale=sigma, size=x.shape[0])
    warp_steps = np.cumsum(random_warp)
    warp_steps = (
        (warp_steps - warp_steps.min())
        / (warp_steps.max() - warp_steps.min())
        * (x.shape[0] - 1)
    )
    cs = CubicSpline(warp_steps, x, axis=0)
    return cs(orig_steps)


def magnitude_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    warp_steps = np.linspace(0, x.shape[0] - 1, num=knot + 2)
    cs = CubicSpline(warp_steps, random_warps)
    return x * cs(orig_steps).reshape(-1, 1)


def window_warp2(x, window_ratio=0.1, scales=[0.5, 2.0]):
    warp_size = int(np.ceil(window_ratio * x.shape[0]))
    start = np.random.randint(0, x.shape[0] - warp_size)
    window = x[start : start + warp_size]
    scale = np.random.choice(scales)
    window = resample(window, int(warp_size * scale))
    warped = np.concatenate((x[:start], window, x[start + warp_size :]))
    return resample(warped, x.shape[0])


def window_warp(x, window_ratio=0.1, scale=1.5):
    """Pick a random window, speed it up or slow it down then reinsert."""
    n = len(x)
    win_len = int(n * window_ratio)
    if win_len < 2:
        return x.copy()
    # pick random window
    start = np.random.randint(0, n - win_len)
    window = x[start : start + win_len]
    # original index and cubic interpolator
    orig_idx = np.arange(win_len)
    cs = CubicSpline(orig_idx, window)
    # warp time axis and re-sample to the same length
    # dividing by scale compresses (speed up), multiplying slows
    warped_idx = np.clip(orig_idx / scale, 0, win_len - 1)
    window_warped = cs(warped_idx)
    # reinsert
    y = x.copy()
    y[start : start + win_len] = window_warped
    return y


"""
import matplotlib.pyplot as plt
import pandas as pd
from behavior import utils as bu

df = pd.read_csv("/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv", header=None)
a = df[df[1]=="2012-05-15 03:15:00"].iloc[:20].copy()
time = a.iloc[:,2].values.copy()
x = a.iloc[:,4:7].values
y = jitter(x, .05)
# tx = torch.tensor(x, dtype=torch.float32)
tx = torch.tensor(a.iloc[:,4:].values.copy(), dtype=torch.float32)
y2 = RandomJitter(0.05)(tx)

y = scaling(x, .05)
y2 = RandomScaling(0.05)(tx)
# y = time_warp(x, .05)
# y = magnitude_warp(x, sigma=0.05, knot=4)
y = window_warp(x)
bu.plot_one(x)
bu.plot_one(y)
print("Done")
"""
