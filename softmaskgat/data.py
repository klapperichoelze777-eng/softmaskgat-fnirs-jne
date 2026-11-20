"""Processed HbO/HbR input, Anchor-Lite features and channel handling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal
from scipy.integrate import trapezoid
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

FS = 11.0
SAMPLES = 991
CHANNELS = 48


def anchor_row(wave):
    """Task statistics (HbO then HbR), followed by task minus baseline."""
    def stats(trace):
        values = np.asarray(trace, dtype=np.float64)
        time = np.arange(values.size, dtype=np.float64) / FS
        return np.asarray([values.mean(), values.std(ddof=1),
                           trapezoid(values, dx=1 / FS),
                           linregress(time, values).slope], dtype=np.float32)

    task = np.concatenate([stats(wave[55:715, chrom]) for chrom in range(2)])
    base = np.concatenate([stats(wave[:55, chrom]) for chrom in range(2)])
    return np.concatenate([task, task - base])


def spectral_ratios(signal, mask, dtype=np.float64):
    """Geometric mean of the two post-filter Welch power ratios."""
    ratios = np.zeros((*mask.shape, 2), dtype=dtype)
    for person, channel in np.argwhere(mask):
        for chrom in range(2):
            trace = signal[person, :, channel, chrom]
            trace = trace - trace.mean()
            if np.allclose(trace, 0):
                continue
            freq, power = sp_signal.welch(trace, fs=FS, nperseg=256, scaling="spectrum")
            inside = (freq >= 0.01) & (freq <= 0.15)
            ratios[person, channel, chrom] = power[inside].sum() / (power[freq > 0.15].sum() + 1e-12)
    return np.sqrt(np.maximum(ratios[..., 0], 0) * np.maximum(ratios[..., 1], 0))


def quality_weights(ratios, mask, fit_rows):
    log_ratios = np.log(ratios + 1e-6)
    valid = mask[fit_rows] & (ratios[fit_rows] > 0)
    sample = log_ratios[fit_rows][valid]
    if not sample.size:
        raise ValueError("No positive spectral ratio on the fitting participants.")
    scaled = (log_ratios - sample.mean()) / (sample.std() + 1e-6)
    with np.errstate(over="ignore"):
        logistic = 1.0 / (1.0 + np.exp(-scaled))
    return np.where(mask, 0.1 + 0.9 * logistic, 0.0).astype(np.float32)


def adjacency(coords):
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((delta ** 2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    edges = np.zeros(distances.shape, dtype=np.float32)
    for channel in range(len(coords)):
        edges[channel, np.argsort(distances[channel])[:6]] = 1
    edges = np.maximum(edges, edges.T)
    np.fill_diagonal(edges, 1)
    return edges


@dataclass
class Dataset:
    signal: np.ndarray
    mask: np.ndarray
    label: np.ndarray
    coords: np.ndarray
    anchor: np.ndarray
    ids: np.ndarray

    @classmethod
    def from_arrays(cls, signal, mask, label, coords, ids, anchor=None):
        signal = np.array(signal, dtype=np.float32, copy=True)
        mask, label, ids = map(np.asarray, (mask, label, ids))
        coords = np.asarray(coords, dtype=np.float64)
        if signal.ndim != 4 or signal.shape[1:] != (SAMPLES, CHANNELS, 2):
            raise ValueError("signal must have shape (participant, 991, 48, 2).")
        n = len(signal)
        if mask.shape != (n, CHANNELS) or not np.isin(mask, [False, True]).all():
            raise ValueError("One binary channel mask per participant is required.")
        mask = mask.astype(bool)
        if not mask.any(axis=1).all():
            raise ValueError("Each participant needs at least one observed channel.")
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("Each row needs a unique participant ID.")
        if label.shape != (n,) or set(label.tolist()) != {0, 1}:
            raise ValueError("One binary label per participant, with both classes represented, is required.")
        if coords.shape != (CHANNELS, 3) or not np.isfinite(coords).all():
            raise ValueError("Supply 48 finite channel-midpoint coordinates in three dimensions.")
        streams = signal.transpose(0, 2, 1, 3)
        streams[~mask] = 0
        if not np.isfinite(streams).all():
            raise ValueError("Observed channel signals must be finite.")
        if anchor is None:
            anchor = np.zeros((n, CHANNELS, 16), dtype=np.float32)
            for i, ch in np.argwhere(mask):
                anchor[i, ch] = anchor_row(streams[i, ch])
        else:
            anchor = np.array(anchor, dtype=np.float32, copy=True)
            if anchor.shape != (n, CHANNELS, 16):
                raise ValueError("anchor must have shape (participant, 48, 16).")
            anchor[~mask] = 0
            if not np.isfinite(anchor).all():
                raise ValueError("Observed Anchor-Lite values must be finite.")
        return cls(signal, mask, label.astype(np.int64), coords, anchor, ids)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls.from_arrays(data["signal"], data["mask"], data["label"],
                                   data["coords"], data["ids"], data.get("anchor"))


def interpolate(data):
    """Repair missing channels from original planar donors with inverse-square weights."""
    xy = data.coords[:, :2]
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("I2 requires unique planar channel coordinates.")
    signal, anchor = data.signal.copy(), data.anchor.copy()
    for person, target in np.argwhere(~data.mask):
        donors = np.flatnonzero(data.mask[person])
        if len(donors) < 4:
            raise ValueError("I2 needs at least four originally observed donor channels.")
        distances = np.linalg.norm(xy[donors] - xy[target], axis=1)
        order = np.lexsort((donors, distances))
        donors, distances = donors[order], distances[order]
        cutoff = distances[3]
        keep = (distances < cutoff) | np.isclose(distances, cutoff, rtol=1e-12, atol=1e-12)
        donors, distances = donors[keep], distances[keep]
        weights = distances ** -2
        weights /= weights.sum()
        for chrom in range(2):
            source = data.signal[person][:, donors, chrom].astype(np.float64)
            signal[person, :, target, chrom] = source @ weights
        anchor[person, target] = anchor_row(signal[person, :, target, :])
    return signal, anchor


def scale_anchor(anchor, mask, train, original_rule=False):
    flat = anchor[train].reshape(-1, 16)
    valid = flat.sum(axis=1) != 0 if original_rule else mask[train].reshape(-1)
    if not valid.any():
        raise ValueError("The training fold has no eligible Anchor-Lite rows.")
    scaler = StandardScaler().fit(flat[valid])
    values = scaler.transform(anchor.reshape(-1, 16)).reshape(anchor.shape)
    values[~mask] = 0
    return values.astype(np.float32)


def synthetic(n=8, seed=7):
    """Generate independent waveform, channel geometry and label examples."""
    if n < 4:
        raise ValueError("Use at least four synthetic participants.")
    rng = np.random.default_rng(seed)
    coords = np.array([(x, y, 0.04 * (x*x + y*y)) for y in range(6) for x in range(8)])
    t = np.arange(SAMPLES) / FS
    phase = rng.uniform(0, 2*np.pi, size=(n, 1, CHANNELS, 2))
    label = np.arange(n) % 2
    signal = np.sin(2 * np.pi * 0.07 * t[None, :, None, None] + phase)
    signal += 0.18 * rng.normal(size=signal.shape)
    mask = rng.uniform(size=(n, CHANNELS)) > 0.12
    mask[:, :4] = True
    mask[:, -1] = False
    return Dataset.from_arrays(signal, mask, label, coords, np.arange(n))
