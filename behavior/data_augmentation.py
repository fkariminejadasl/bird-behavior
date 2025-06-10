import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomJitter:
    """Add element-wise Gaussian noise N(0, σ²) to the first 3 dims;
    to the 4th dim, apply one single random offset (or zero if you prefer)."""

    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        noise = torch.randn_like(x) * self.sigma
        if x.size(1) == 4:
            # draw one scalar noise for the last column
            # last_noise = torch.randn((), device=x.device, dtype=x.dtype) * self.sigma
            # noise[:, 3] = last_noise
            noise[:, 3] = 0  # keep the last channel unchanged
        return x + noise


class RandomScaling:
    """Scale each feature channel by a factor ~ N(1, σ²)."""

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D)
        scales = (
            torch.randn(x.size(1), device=x.device, dtype=x.dtype) * self.sigma + 1.0
        )
        if x.size(1) == 4:
            scales[3] = 1.0  # keep the last channel unchanged
        return x * scales.unsqueeze(0)


class MagnitudeWarp:
    """Apply magnitude warping to first 3 dims and Gaussian jitter to the 4th dim when C==4."""

    def __init__(self, sigma: float = 0.2, knot: int = 4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a magnitude warp to a 2D time-series tensor via cubic interpolation.

        Args:
            x      (Tensor): shape (seq_len, channels) [T, C]
            sigma  (float): std. dev. of the random warping factors (around 1.0)
            knot   (int):   number of internal knots (the total control points will be knot+2)

        Returns:
            Tensor of same shape as x, warped in magnitude.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected (T, C) but got {tuple(x.shape)}")

        T, C = x.shape
        device, dtype = x.device, x.dtype

        # 1) build one shared warp curve of shape [T]
        cps = torch.normal(
            1.0, self.sigma, size=(1, 1, 1, self.knot + 2), device=device, dtype=dtype
        )
        warp = F.interpolate(cps, size=(1, T), mode="bicubic", align_corners=True)
        warp = warp.view(1, T)  # [1, T]

        # 2) if 4 channels, split; else, warp everything
        if C == 4:
            # warp first 3 dims
            x3 = x[:, :3].t().contiguous()  # [3, T]
            x3_warp = (x3 * warp).permute(1, 0)  # [T, 3]

            # jitter last dim independently per time step
            noise4 = torch.randn((), device=device, dtype=dtype) * self.sigma
            x4_jit = x[:, 3] + noise4  # [T]

            return torch.cat([x3_warp, x4_jit.unsqueeze(1)], dim=1)

        # otherwise warp all C channels
        x_t = x.t().contiguous()  # [C, T]
        x_all_w = (x_t * warp).permute(1, 0)  # [T, C]
        return x_all_w


def magnitude_warp_torch(
    x: torch.Tensor, sigma: float = 0.2, knot: int = 4
) -> torch.Tensor:
    """
    Applies a magnitude warp to a 2D time-series tensor via cubic interpolation.

    Args:
        x      (Tensor): shape (seq_len, channels) [T, C]
        sigma  (float): std. dev. of the random warping factors (around 1.0)
        knot   (int):   number of internal knots (the total control points will be knot+2)

    Returns:
        Tensor of same shape as x, warped in magnitude.
    """
    if x.dim() != 2:
        raise ValueError("Expected x of shape (T, C)")

    T, C = x.shape

    # sample random warp control points (shape: [B=1, 1, 1, knot+2])
    device = x.device
    cps = torch.normal(1.0, sigma, size=(1, 1, 1, knot + 2), device=device)

    # bicubically interpolate from width=knot+2 → width=T
    # treat H=1, W=knots+2 → H=1, W=T
    # warp shape: [B=1, 1, 1, knot+2]
    warp = F.interpolate(cps, size=(1, T), mode="bicubic", align_corners=True)
    warp = warp.view(1, T)  # [1, T]

    # Apply warp to each channel (broadcasting over C)
    x_ = x.t().contiguous()
    x_warped = x_ * warp  # [C, T]
    x_warped = x_warped.permute(1, 0).contiguous()  # [T, C]

    return x_warped


class TimeWarp:
    def __init__(self, sigma: float = 0.2):

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a time warp to a 2D time-series tensor via linear interpolation.

        Args:
            x      (Tensor): shape (seq_len, channels) [T, C]
            sigma  (float): std. dev. of the random warping factors (around 1.0)

        Returns:
            Tensor of same shape as x, warped in time dimension.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x of shape (T, C), got {x.shape}")

        T, C = x.shape
        device, dtype = x.device, x.dtype

        # 1) generate a warp curve of length T
        rw = torch.normal(1.0, self.sigma, size=(T,), device=device, dtype=dtype)
        warp_steps = torch.cumsum(rw, dim=0)
        warp_steps = (
            (warp_steps - warp_steps.min())
            / (warp_steps.max() - warp_steps.min())
            * (T - 1)
        )

        # normalize to [-1,1] for grid_sample
        warp_norm = 2.0 * warp_steps / (T - 1) - 1.0  # shape: [T]

        # build sampling grid: [1, 1, T, 2]; X=warp_norm, Y=0
        sample_grid = torch.zeros(1, 1, T, 2, device=device, dtype=dtype)
        sample_grid[..., 0] = warp_norm.view(1, 1, T)  # time coords
        # sample_grid[..., 1] stays zero (height coord)

        # helper to grid-sample any (batch=1,H=1,W=T) input
        def apply_warp(inp: torch.Tensor) -> torch.Tensor:
            # inp: [T, c] → [1, c, 1, T]
            inp4 = inp.t().view(1, inp.shape[1], 1, T)
            out4 = F.grid_sample(
                inp4,
                sample_grid,
                mode="bicubic",
                padding_mode="border",
                align_corners=True,
            )
            # → [1, c, 1, T] → [T, c]
            return out4.squeeze(2).squeeze(0).t()

        if C == 4:
            # warp first 3 channels
            x3 = x[:, :3]  # [T,3]
            x3_w = apply_warp(x3)  # [T,3]

            # jitter 4th channel
            noise4 = torch.randn((), device=device, dtype=dtype) * self.sigma
            x4_j = x[:, 3] + noise4  # [T]

            return torch.cat([x3_w, x4_j.unsqueeze(1)], dim=1)

        # else: warp all channels
        return apply_warp(x)
        # # Linear interp indices
        # idx0 = torch.floor(cum).long().clamp(0, T-2)
        # idx1 = idx0 + 1
        # x0, x1 = x[idx0], x[idx1] # (T,C)
        # frac = (cum - idx0.float()).unsqueeze(1) # (T,1)
        # return x0 * (1 - frac) + x1 * frac  # (T,C)


def time_warp_torch(x: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    """
    Applies a time warp to a 2D time-series tensor via linear interpolation.

    Args:
        x      (Tensor): shape (seq_len, channels) [T, C]
        sigma  (float): std. dev. of the random warping factors (around 1.0)

    Returns:
        Tensor of same shape as x, warped in time dimension.
    """
    if x.dim() != 2:
        raise ValueError("Expected x of shape (T, C)")

    T, C = x.shape
    device = x.device

    # Generate random warping factors
    random_warp = torch.normal(1.0, sigma, size=(T,), device=device)

    # Create cumulative sum to generate warped time steps
    warp_steps = torch.cumsum(random_warp, dim=0)

    # Normalize to range [0, T-1]
    warp_steps = (
        (warp_steps - warp_steps.min())
        / (warp_steps.max() - warp_steps.min())
        * (T - 1)
    )

    # Prepare x for grid_sample: [batch=1, channels=C, height=1, width=T]
    x_4d = x.t().contiguous().view(1, C, 1, T)

    # First, create a grid that maps from original to warped positions
    warp_norm = 2.0 * warp_steps / (T - 1) - 1.0  # Normalize to [-1, 1]

    # Create sampling grid
    sample_grid = torch.zeros(1, 1, T, 2, device=device)
    sample_grid[0, 0, :, 0] = warp_norm  # X coordinates (time)
    # Y coordinates remain 0

    # Use grid_sample for the warping
    x_warped = F.grid_sample(
        x_4d, sample_grid, mode="bicubic", padding_mode="border", align_corners=True
    )

    # Reshape back to [T, C]
    x_warped = x_warped.squeeze(2).squeeze(0).t()

    return x_warped
    # # Linear interp indices
    # idx0 = torch.floor(cum).long().clamp(0, T-2)
    # idx1 = idx0 + 1
    # x0, x1 = x[idx0], x[idx1] # (T,C)
    # frac = (cum - idx0.float()).unsqueeze(1) # (T,1)
    # return x0 * (1 - frac) + x1 * frac  # (T,C)


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


def magnitude_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    warp_steps = np.linspace(0, x.shape[0] - 1, num=knot + 2)
    cs = CubicSpline(warp_steps, random_warps)
    return x * cs(orig_steps).reshape(-1, 1)


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

df = pd.read_csv(
    "/home/fatemeh/Downloads/bird/data/final/proc2/starts.csv", header=None
)
a = df[df[1] == "2012-05-15 03:15:00"].iloc[:20].copy()
time = a.iloc[:, 2].values.copy()
x = a.iloc[:, 4:7].values
tx = torch.tensor(x, dtype=torch.float32)
# tx = torch.tensor(a.iloc[:,4:].values.copy(), dtype=torch.float32)

y = time_warp(x, sigma=0.05)
y2 = time_warp_torch(tx, sigma=0.05)

# y = magnitude_warp(x, sigma=0.05, knot=4)
# y2 = magnitude_warp_torch(tx, sigma=0.05, knot=4)

# y2 = RandomJitter(0.05)(tx)
# y2 = RandomScaling(0.05)(tx)

# y = jitter(x, 0.05)
# y = scaling(x, .05)
# y = time_warp(x, .05)
# y = magnitude_warp(x, sigma=0.05, knot=4)
# y = window_warp(x)
bu.plot_one(x)
bu.plot_one(y)
# bu.plot_one(np.array(y2))
print("Done")
"""
