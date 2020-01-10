from __future__ import annotations

import functools
import math
from typing import List, Tuple

import numpy as np

from src.layouts import TensorLayout, TiledArrayLayout


def tile(
    arr: np.ndarray, in_layout: TensorLayout, out_layout: TiledArrayLayout
) -> np.ndarray:
    """Tiles tensor into 2D image.

    Args:
        arr: tensor
        in_layout: layout of input tensor
        out_layout: layout of output tiled array

    Returns:
        np.ndarray: tiled array
    """
    assert in_layout.shape == out_layout.orig_shape_in_order(in_layout.order)
    assert arr.shape == in_layout.shape
    arr = _as_chw(arr, in_layout.order)
    arr = _tile_chw(arr, out_layout.nrows, out_layout.ncols)
    assert arr.shape == out_layout.shape
    return arr


def _tile_chw(arr: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    """
    Args:
        arr: chw tensor
        nrows: number of tiled rows
        ncols: number of tiled columns

    Returns:
        np.ndarray: tiled array
    """
    c, h, w = arr.shape
    assert c <= nrows * ncols

    if c < nrows * ncols:
        arr = arr.reshape(-1).copy()
        arr.resize(nrows * ncols * h * w)

    return (
        arr.reshape(nrows, ncols, h, w)
        .swapaxes(1, 2)
        .reshape(nrows * h, ncols * w)
    )


def detile(
    arr: np.ndarray, in_layout: TiledArrayLayout, out_layout: TensorLayout
) -> np.ndarray:
    """Detiles 2D image into tensor.

    Args:
        arr: tiled array
        in_layout: layout of input tiled array
        out_layout: layout of output tensor

    Returns:
        np.ndarray: tensor
    """
    assert out_layout.shape == in_layout.orig_shape_in_order(out_layout.order)
    print(arr.shape, in_layout.shape)
    assert arr.shape == in_layout.shape
    arr = _detile_chw(arr, **in_layout.__dict__)
    arr = _as_order(arr, "chw", out_layout.order)
    assert arr.shape == out_layout.shape
    return arr


def _detile_chw(
    arr: np.ndarray, c: int, h: int, w: int, nrows: int, ncols: int, **kwargs
) -> np.ndarray:
    """
    Args:
        arr: tiled array
        c: channels (number of tiles to keep)
        h: height of tile
        w: width of tile
        nrows: number of tiled rows
        ncols: number of tiled columns

    Returns:
        np.ndarray: chw tensor
    """
    return (
        arr.reshape(nrows, h, ncols, w)
        .swapaxes(1, 2)
        .reshape(-1)[: c * h * w]
        .reshape(c, h, w)
    )


def _as_order(arr: np.ndarray, in_order: str, out_order: str) -> np.ndarray:
    if in_order == out_order:
        return arr
    if out_order == "chw":
        return _as_chw(arr, in_order)
    if out_order == "hwc":
        return _as_hwc(arr, in_order)
    raise ValueError(f"Cannot convert to {out_order} from {in_order}")


def _as_chw(arr: np.ndarray, order: str) -> np.ndarray:
    if order == "chw":
        return arr
    if order == "hwc":
        return np.rollaxis(arr, 2, 0)
    raise ValueError(f"Cannot convert to chw from {order}")


def _as_hwc(arr: np.ndarray, order: str) -> np.ndarray:
    if order == "chw":
        return np.rollaxis(arr, 0, 3)
    if order == "hwc":
        return arr
    raise ValueError(f"Cannot convert to hwc from {order}")


def determine_tile_layout(tensor_layout: TensorLayout) -> TiledArrayLayout:
    """Determine reasonable tile layout from given tensor layout."""
    c, h, w = tensor_layout.shape_in_order("chw")
    # nrows = np.product([p ** (k // 2) for p, k in factorize(c)])
    nrows = math.ceil(math.sqrt(c))
    ncols = math.ceil(c / nrows)
    return TiledArrayLayout(tensor_layout.dtype, c, h, w, nrows, ncols)


@functools.lru_cache()
def _primes(n: int) -> List[int]:
    xs = [2]
    for i in range(3, n, 2):
        if all(i % x != 0 for x in xs):
            xs.append(i)
    return xs


@functools.lru_cache()
def _factorize(n: int) -> List[Tuple[int, int]]:
    ps = _primes(int(math.sqrt(n)))
    xs = []
    for p in ps:
        count = 0
        while n % p == 0:
            n /= p
            count += 1
        xs.append((p, count))
    return xs
