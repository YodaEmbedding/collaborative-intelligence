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

from src.layouts import TensorLayout, TiledArrayLayout
from src.modelconfig import ModelConfig, PostencoderConfig
from src.tile import determine_tile_layout, detile


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
        img = np.array(img)[..., 0]
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


def get_predecoder(
    postencoder_config: PostencoderConfig,
    model_config: ModelConfig,
    tensor_layout: TensorLayout,
) -> Predecoder:
    encoder_type = model_config.encoder
    postencoder_type = postencoder_config.type

    if model_config.layer == "client":
        assert encoder_type == "None"
        assert postencoder_type == "None"
        shape = (-1,)
        dtype = np.float32
        return TensorPredecoder(shape, dtype)

    shape = tensor_layout.shape
    dtype = tensor_layout.dtype

    if model_config.layer == "server":
        assert encoder_type == "None"
        if postencoder_type == "None":
            assert shape[-1] == 3
            return RgbPredecoder(shape, dtype)
        if postencoder_type == "jpeg":
            return JpegRgbPredecoder(tensor_layout)
        raise ValueError("Unknown postencoder")

    if encoder_type == "None":
        assert postencoder_type == "None"
        return TensorPredecoder(shape, dtype)

    if encoder_type == "UniformQuantizationU8Encoder":
        assert dtype == np.uint8
        if postencoder_type == "None":
            return TensorPredecoder(shape, dtype)
        if postencoder_type == "jpeg":
            tiled_layout = determine_tile_layout(tensor_layout)
            return JpegPredecoder(tiled_layout, tensor_layout)
        raise ValueError("Unknown postencoder")

    raise ValueError("Unknown encoder")


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
