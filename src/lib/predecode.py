from io import BytesIO
from math import ceil
from typing import (
    Awaitable,
    ByteString,
    Dict,
    Generator,
    Generic,
    List,
    Tuple,
    TypeVar,
)

import numpy as np
import tensorflow as tf
from PIL import Image

from src.lib.layouts import TensorLayout, TiledArrayLayout
from src.lib.tile import detile


class Predecoder:
    def run(self, buf: ByteString) -> np.ndarray:
        raise NotImplementedError


class TensorPredecoder(Predecoder):
    def __init__(self, shape: tuple, dtype: type):
        self._shape = shape
        self._dtype = to_np_dtype(dtype)

    def run(self, buf: ByteString) -> np.ndarray:
        return np.frombuffer(buf, dtype=self._dtype).reshape(self._shape)


class RgbPredecoder(Predecoder):
    def __init__(self, shape: tuple, dtype: type):
        self._shape = shape
        self._dtype = to_np_dtype(dtype)

    def run(self, buf: ByteString) -> np.ndarray:
        return (
            np.frombuffer(buf, dtype=np.uint8)
            .reshape(self._shape)
            .astype(self._dtype)
        )


class JpegPredecoder(Predecoder):
    MBU_SIZE = 16

    def __init__(
        self, tiled_layout: TiledArrayLayout, tensor_layout: TensorLayout
    ):
        self._tiled_layout = tiled_layout
        self._tensor_layout = tensor_layout

    def run(self, buf: ByteString) -> np.ndarray:
        img = _decode_raw_img(buf)
        img = np.array(img)
        img = self._trim(img)
        tensor = detile(img, self._tiled_layout, self._tensor_layout)
        return tensor

    def _trim(self, img: np.ndarray) -> np.ndarray:
        shape = self._tiled_layout.shape
        expect = tuple(ceil(x / self.MBU_SIZE) * self.MBU_SIZE for x in shape)
        assert expect == img.shape
        return img[: shape[0], : shape[1]]


class JpegRgbPredecoder(Predecoder):
    def __init__(self, tensor_layout: TensorLayout):
        self._tensor_layout = tensor_layout

    def run(self, buf: ByteString) -> np.ndarray:
        img = _decode_raw_img(buf)
        return np.array(img).astype(self._tensor_layout.dtype)


def to_np_dtype(dtype: type) -> type:
    return {
        "float32": np.float32,
        "uint8": np.uint8,
        np.float32: np.float32,
        np.uint8: np.uint8,
        tf.float32: np.float32,
        tf.uint8: np.uint8,
    }[dtype]


def _decode_raw_img(buf: ByteString) -> Image.Image:
    with BytesIO(buf) as stream:
        img = Image.open(stream)
        img.load()
    return img
