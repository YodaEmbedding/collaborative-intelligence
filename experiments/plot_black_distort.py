from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import plot


def random_mask(pct: float, shape: Tuple[int]) -> np.ndarray:
    """Returns mask of given shape with pct of values set to True."""
    size = np.prod(shape)
    n = int(pct * size)
    if np.random.rand() < pct * size - n:
        n = n + 1
    idxs = np.random.choice(size, n, replace=False)
    mask = np.zeros(shape, dtype=np.bool)
    mask.reshape(-1)[idxs] = True
    return mask


def black_neuron_pr(x: np.ndarray, black: np.ndarray, p) -> np.ndarray:
    mask = random_mask(p, x.shape)
    x = x.copy()
    x[mask] = black[mask]
    return x


def black_channel_pr(x: np.ndarray, black: np.ndarray, p) -> np.ndarray:
    mask = random_mask(p, (x.shape[-1],))
    x = x.copy()
    x[..., mask] = black[..., mask]
    return x


basename = "img/experiment/distort_tensor"


def plot_quick(plot_func, tensor, suffix, **kwargs):
    fig = plot_func(tensor, "", **kwargs)
    plot.save(fig, f"{basename}-{suffix}.png", bbox_inches="tight")


x = np.load("img/experiment/sample.npy")
mean = np.mean(x)
std = np.std(x)
sigma = 2.5
x = np.clip(x, mean - sigma * std, mean + sigma * std)
black = np.ones_like(x) * x.min()
p = 0.4

y = black_neuron_pr(x, black, p)
plot_quick(plot.featuremap, y, "blacken_neuron")

y = black_channel_pr(x, black, p)
plot_quick(plot.featuremap, y, "blacken_channel")
