from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

# TODO also include dtype information in layouts?


@dataclass(eq=True, frozen=True)
class Layout:
    dtype: type

    @property
    def shape(self) -> tuple:
        raise NotImplementedError


@dataclass(eq=True, frozen=True)
class _ChwMixin(Layout):
    c: int
    h: int
    w: int
    order: str

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.shape_in_order(self.order)

    def shape_in_order(self, order: str) -> Tuple[int, int, int]:
        return _shape_in_order(self.__dict__, order)


@dataclass(eq=True, frozen=True)
class TensorLayout(_ChwMixin):
    @staticmethod
    def from_shape(
        shape: Tuple[int, int, int], order: str, dtype: type
    ) -> TensorLayout:
        return TensorLayout(dtype, **_from_shape(shape, order))

    @staticmethod
    def from_tensor(tensor: np.ndarray, order: str) -> TensorLayout:
        return TensorLayout(tensor.dtype, **_from_shape(tensor.shape, order))


@dataclass(eq=True, frozen=True)
class RgbLayout(_ChwMixin):
    def __post_init__(self):
        assert self.c == 3

    @staticmethod
    def from_shape(
        shape: Tuple[int, int, int], order: str, dtype: type
    ) -> RgbLayout:
        return RgbLayout(dtype, **_from_shape(shape, order))


@dataclass(eq=True, frozen=True)
class TiledArrayLayout(Layout):
    c: int
    h: int
    w: int
    nrows: int
    ncols: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nrows * self.h, self.ncols * self.w)

    def orig_shape_in_order(self, order: str) -> Tuple[int, int, int]:
        return _shape_in_order(self.__dict__, order)


def _from_shape(shape: Tuple[int, int, int], order: str) -> Dict[str, Any]:
    assert len(order) == 3
    kwargs = {k: shape[i] for i, k in enumerate(order)}
    kwargs["order"] = order
    return kwargs


def _shape_in_order(d: Dict[str, int], order: str) -> Tuple[int, int, int]:
    assert len(order) == 3
    return tuple(d[x] for x in order)
