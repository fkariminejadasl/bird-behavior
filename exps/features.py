"""
from: Jasper A. J. Eikelboom, 2026, Optimal deep learning activity recognition pipeline using animal bio-logging data

Summary features:
0th, 1st, 2nd, 3rd, 4th quartile
mean
standard deviation
skewness
kurtosis

Sequential features:
ACF lag 1
ACF lag 1 of first differences
standard deviation of first differences
number of mean crossings
average rectified value

# Frequency-domain features
1st to 5th dominant amplitude
frequency of 1st to 5th dominant amplitude
spectral energy
entropy

For L=20, C=3:
quartiles:       5 x 3 = 15
mean:            1 x 3 = 3
std:             1 x 3 = 3
skewness:        1 x 3 = 3
kurtosis:        1 x 3 = 3
summary total:   9 x 3 = 27 features

sequential total: 5 x 3 = 15 features

5 amplitudes + 5 frequencies + 1 energy + 1 entropy = 12 features per channel
frequency total: 12 x 3 = 36 features


9C + 5C + 12C = 26C = 78 features for C=3


Judy features:
from: Shamoun-Baranes et al, 2016, Flap or soar? How a flight generalist responds to its aerial environment
==============
mean_x/y/z (B, 3): Static body orientation and posture.
std_x/y/z (B, 3): Movement amplitude, stronger during active/flapping behavior.
mean_pitch, std_pitch (B, 2): Body angle and angle variability.
mean_roll, std_roll (B, 2): Side tilt and tilt variability.
correlation_xy/yz/xz (B, 3): Coupling between axes, useful for periodic wingbeat patterns.
gps_speed (B, 1): Helps separate stationary, floating, boat, walking, and flight.
meanabsder_x/y/z (B, 3): Mean absolute change over time, useful for intensity and wingbeat motion.
noise_x/y/z (B, 3): High-frequency irregularity, useful for chaotic or non-smooth motion.
noise/absder_x/y/z (B, 3): Noise normalized by movement intensity.
fundfreq_x/y/z (B, 3): Dominant frequency, useful for wingbeat or walking cadence.
odba, vedba (B, 2): Dynamic body acceleration, often used as movement or energy proxy.
fundfreqcorr_x/y/z (B, 3): How close the signal is to a clean periodic oscillation.
fundfreqmagnitude_x/y/z (B, 3): Strength of dominant periodic motion.
first_x/y/z (B, 3): Initial axis values, mostly posture/orientation context.

37 features (B, 37) in total, or 14 selected features (B, 14) if selected=True.
"""

import math

import torch


def summary_features(x: torch.Tensor, eps: float = 1e-8):
    # x: (B, L, C)
    B, L, C = x.shape

    q_levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=x.device)
    q = torch.quantile(x, q_levels, dim=1).permute(1, 0, 2)
    # q: (B, 5, C)

    mean = x.mean(dim=1, keepdim=True)
    # mean: (B, 1, C)

    std = x.std(dim=1, correction=1, keepdim=True)
    # std: (B, 1, C)

    z = (x - mean) / (std + eps)

    skew = (z**3).mean(dim=1, keepdim=True)
    # skew: (B, 1, C)

    kurt = (z**4).mean(dim=1, keepdim=True)
    # kurt: (B, 1, C)
    # Note: this is raw kurtosis, not excess kurtosis.

    feats = torch.cat([q, mean, std, skew, kurt], dim=1)
    # feats: (B, 9, C)

    return feats.reshape(B, 9 * C)
    # output: (B, 27) when C=3


def acf_lag1(y: torch.Tensor, eps: float = 1e-8):
    # y: (B, T, C)
    y0 = y[:, :-1, :]
    y1 = y[:, 1:, :]

    mu = y.mean(dim=1, keepdim=True)

    num = ((y0 - mu) * (y1 - mu)).sum(dim=1)
    den = ((y - mu) ** 2).sum(dim=1) + eps

    return num / den
    # output: (B, C)


def sequential_features(x: torch.Tensor, eps: float = 1e-8):
    # x: (B, L, C)
    B, L, C = x.shape

    mean = x.mean(dim=1, keepdim=True)

    diff = x[:, 1:, :] - x[:, :-1, :]
    # diff: (B, L-1, C)

    acf1 = acf_lag1(x, eps).unsqueeze(1)
    # (B, 1, C)

    acf1_diff = acf_lag1(diff, eps).unsqueeze(1)
    # (B, 1, C)

    std_diff = diff.std(dim=1, correction=1, keepdim=True)
    # (B, 1, C)

    mean_cross = (
        (((x[:, :-1, :] - mean) * (x[:, 1:, :] - mean)) < 0)
        .sum(dim=1, keepdim=True)
        .float()
    )
    # (B, 1, C)

    arv = x.abs().mean(dim=1, keepdim=True)
    # average rectified value: (B, 1, C)

    feats = torch.cat([acf1, acf1_diff, std_diff, mean_cross, arv], dim=1)
    # feats: (B, 5, C)

    return feats.reshape(B, 5 * C)
    # output: (B, 15) when C=3


def fft_features(x: torch.Tensor, dt: float = 1.0, eps: float = 1e-8):
    # x: (B, L, C)
    B, L, C = x.shape

    X = torch.fft.rfft(x, dim=1)
    mag = X.abs()
    # mag: (B, F, C)

    freqs = torch.fft.rfftfreq(L, d=dt).to(x.device)
    # freqs: (F,)

    # usually exclude DC component for dominant movement frequencies
    mag_no_dc = mag[:, 1:, :]
    freqs_no_dc = freqs[1:]

    k = min(5, mag_no_dc.shape[1])

    top_amp, top_idx = torch.topk(mag_no_dc, k=k, dim=1)
    # top_amp: (B, 5, C), if enough frequencies

    top_freq = freqs_no_dc[top_idx]
    # top_freq: (B, 5, C)

    # pad if L is very small and fewer than 5 frequencies exist
    if k < 5:
        pad_amp = torch.zeros(B, 5 - k, C, device=x.device)
        pad_freq = torch.zeros(B, 5 - k, C, device=x.device)
        top_amp = torch.cat([top_amp, pad_amp], dim=1)
        top_freq = torch.cat([top_freq, pad_freq], dim=1)

    spectral_energy = (mag_no_dc**2).sum(dim=1, keepdim=True)
    # (B, 1, C)

    p = mag_no_dc / (mag_no_dc.sum(dim=1, keepdim=True) + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)
    # (B, 1, C)

    feats = torch.cat([top_amp, top_freq, spectral_energy, entropy], dim=1)
    # feats: (B, 12, C)

    return feats.reshape(B, 12 * C)
    # output: (B, 36) when C=3


def corr(a, b, eps=1e-8):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / ((a * a).sum(1).sqrt() * (b * b).sum(1).sqrt() + eps)


def judy_features(acc, gps_speed, fs=20.0, eps=1e-8, selected=False):
    B, L, _ = acc.shape
    x, y, z = acc[..., 0], acc[..., 1], acc[..., 2]

    mean_xyz = acc.mean(1)
    std_xyz = acc.std(1)

    pitch = torch.atan2(x, (y * y + z * z + eps).sqrt()) * 180 / math.pi
    roll = torch.atan2(y, (x * x + z * z + eps).sqrt()) * 180 / math.pi

    pitch_roll = torch.stack(
        [pitch.mean(1), pitch.std(1), roll.mean(1), roll.std(1)], 1
    )

    corrs = torch.stack([corr(x, y), corr(y, z), corr(x, z)], 1)

    der = (acc[:, 1:] - acc[:, :-1]).abs().mean(1) * fs

    noise_sig = (-0.5 * acc[:, :-2] + acc[:, 1:-1] - 0.5 * acc[:, 2:]).abs().mean(1)
    noise_der = noise_sig / (der + eps)

    dba = acc - acc.mean(1, keepdim=True)
    odba = dba.abs().sum(2).mean(1, keepdim=True)
    vedba = (dba * dba).sum(2).sqrt().mean(1, keepdim=True)

    X = torch.fft.rfft(acc - acc.mean(1, keepdim=True), dim=1).abs()
    freqs = torch.fft.rfftfreq(L, d=1 / fs).to(acc.device)
    X1 = X[:, 1:]
    idx = X1.argmax(1)
    fundfreq = freqs[1:][idx]
    fundmag = torch.gather(X1, 1, idx[:, None, :]).squeeze(1)

    t = torch.arange(L, device=acc.device).float() / fs
    phase = 2 * math.pi * fundfreq[:, None, :] * t[None, :, None]
    sine_corr = torch.stack(
        [
            corr(x, torch.sin(phase[..., 0])),
            corr(y, torch.sin(phase[..., 1])),
            corr(z, torch.sin(phase[..., 2])),
        ],
        1,
    )

    first_xyz = acc[:, 0]

    f = torch.cat(
        [
            mean_xyz,
            std_xyz,
            pitch_roll,
            corrs,
            gps_speed.reshape(B, 1),
            der,
            noise_sig,
            noise_der,
            fundfreq,
            odba,
            vedba,
            sine_corr,
            fundmag,
            first_xyz,
        ],
        1,
    )

    sel = [0, 3, 5, 6, 13, 16, 17, 18, 22, 23, 26, 27, 31, 33]
    return f[:, sel] if selected else f
