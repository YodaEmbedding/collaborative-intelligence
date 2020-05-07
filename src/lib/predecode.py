from io import BytesIO
from math import ceil
from typing import ByteString, Callable

import numpy as np
import tensorflow as tf
from PIL import Image

from src.lib.layouts import TensorLayout, TiledArrayLayout
from src.lib.tile import detile


class Predecoder:
    def run(self, buf: ByteString) -> np.ndarray:
        raise NotImplementedError


class CallablePredecoder(Predecoder):
    def __init__(self, func: Callable[[ByteString], np.ndarray]):
        self.func = func

    def run(self, buf: ByteString) -> np.ndarray:
        return self.func(buf)


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


class _ImageRgbPredecoder(Predecoder):
    def __init__(self, tensor_layout: TensorLayout):
        self._tensor_layout = tensor_layout

    def run(self, buf: ByteString) -> np.ndarray:
        img = _decode_raw_img(buf)
        return np.array(img).astype(self._tensor_layout.dtype)


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
        img = _trim(img, self._tiled_layout, self.MBU_SIZE)
        tensor = detile(img, self._tiled_layout, self._tensor_layout)
        return tensor


class JpegRgbPredecoder(_ImageRgbPredecoder):
    pass


class Jpeg2000Predecoder(Predecoder):
    MBU_SIZE = 16

    def __init__(
        self, tiled_layout: TiledArrayLayout, tensor_layout: TensorLayout
    ):
        self._tiled_layout = tiled_layout
        self._tensor_layout = tensor_layout

    def run(self, buf: ByteString) -> np.ndarray:
        img = _decode_raw_img(buf)
        img = np.array(img)
        img = _trim(img, self._tiled_layout, self.MBU_SIZE)
        tensor = detile(img, self._tiled_layout, self._tensor_layout)
        return tensor


class Jpeg2000RgbPredecoder(_ImageRgbPredecoder):
    pass


class PngPredecoder(Predecoder):
    def __init__(
        self, tiled_layout: TiledArrayLayout, tensor_layout: TensorLayout
    ):
        self._tiled_layout = tiled_layout
        self._tensor_layout = tensor_layout

    def run(self, buf: ByteString) -> np.ndarray:
        img = _decode_raw_img(buf)
        img = np.array(img)
        assert self._tiled_layout.shape == img.shape
        tensor = detile(img, self._tiled_layout, self._tensor_layout)
        return tensor


class PngRgbPredecoder(_ImageRgbPredecoder):
    pass


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


def _trim(
    img: np.ndarray, tiled_layout: TiledArrayLayout, mbu_size: int
) -> np.ndarray:
    shape = tiled_layout.shape
    expect = tuple(ceil(x / mbu_size) * mbu_size for x in shape)
    assert expect == img.shape
    return img[: shape[0], : shape[1]]
