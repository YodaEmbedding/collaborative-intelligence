from typing import Any, Dict, Tuple

import numpy as np


def uni_quant(
    x: np.ndarray, clip_range: Tuple[float, float], levels: int
) -> np.ndarray:
    x = x.astype(np.float32)
    x -= clip_range[0]
    x *= levels / (clip_range[1] - clip_range[0])
    x = np.clip(x, 0, levels - 1)
    x = x.astype(np.uint8)
    return x


def uni_dequant(
    x: np.ndarray, clip_range: Tuple[float, float], levels: int
) -> np.ndarray:
    width = (clip_range[1] - clip_range[0]) / levels
    x = x.astype(np.float32)
    x = np.clip(x, 0, levels - 1)
    x *= width
    x += 0.5 * width + clip_range[0]
    return x


def bin_quant(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x = np.digitize(x, bins).clip(1, len(bins) - 1) - 1
    x = x.astype(np.uint8)
    return x


# TODO specify different reconstruction than midpoint of bin edges
def bin_dequant(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    midpoints = 0.5 * (bins[1:] + bins[:-1])
    return midpoints[x]


def indep_quant(
    x: np.ndarray, d: Dict[str, Any], wstd: float, levels: int
) -> np.ndarray:
    m = d["tensors_mean"]
    s = d["tensors_std"]
    # w = d["tensors_levels"]  # how important neurons are -> give more levels
    w = wstd * s
    x0 = m - w
    x = levels * (x - x0) / (2 * w)
    x = np.clip(x, 0, levels - 1)
    x = x.astype(np.uint8)
    return x


def indep_dequant(
    x: np.ndarray, d: Dict[str, Any], wstd: float, levels: int
) -> np.ndarray:
    m = d["tensors_mean"]
    s = d["tensors_std"]
    w = wstd * s
    x0 = m - w
    width = (2 * wstd * s) / levels
    x = x.astype(np.float32)
    x = (x + 0.5) * width + x0
    return x


def qcut_bins(x: np.ndarray, levels: int) -> np.ndarray:
    x = np.sort(x.reshape(-1))
    idx = np.linspace(0, len(x) - 1, levels + 1, dtype=int)
    bins = x[idx]

    for i in range(levels):
        if bins[i] < bins[i + 1]:
            continue
        bins[i + 1] = _nextfloat(bins[i], 1)

    unique_bins = np.unique(bins)

    if len(unique_bins) != levels + 1:
        raise ValueError(f"Could not create {levels} bins.\nbins: {bins}")

    return bins


def _nextfloat(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return x + 2 ** -4 * np.copysign(x, direction)
    # return np.nextafter(x, x + direction)
