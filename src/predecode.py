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
from tensorflow import keras
from tensorflow.keras.applications import imagenet_utils

from src.layers import decoders
from src.modelconfig import ModelConfig
from src.tile import (
    TensorLayout,
    TiledArrayLayout,
    determine_tile_layout,
    detile,
)


class Predecoder:
    def run(self, buf: ByteString) -> np.ndarray:
        raise NotImplementedError


class SimplePredecoder(Predecoder):
    # def __init__(self, tensor_layout: TensorLayout):
    #     self._tensor_layout = tensor_layout

    def __init__(self, shape: tuple, dtype: type):
        self._shape = shape
        self._dtype = to_np_dtype(dtype)

    def run(self, buf: ByteString) -> np.ndarray:
        return np.frombuffer(buf, dtype=self._dtype).reshape(
            (-1, *self._shape)
        )

        # shape = self._tensor_layout.shape

        # RGB input is also in hwc, btw... no need to make exception?
        # also, rgb input JPEG isn't tiled, so...
        # tiled_layout = TiledLayout(3, 224, 224, "hwc") # chw?
        # but then, we shouldn't uhhh... change chw order? idk
        # also, need JpegEncoder on client to encode RGB instead of custom YUV strategy
        # Tiled/TensorLayout is perhaps a subclass of other Layouts (also, RgbLayout?)

        # perhaps layouts should also include their dtypes?

        # THEN WE JUST NEED TO DEFINE CONVERSIONS BETWEEN _LAYOUTS_:
        # convert(data, in_layout, out_layout)

        # return np.frombuffer(buf, dtype=np.float32)[None, ...]
        # input_type = model.layers[0].dtype
        # dtype = to_np_dtype(input_type)
        # return np.frombuffer(buf, dtype=dtype).reshape((-1, *shape))


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

        # TODO extract detiling from predecoder
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
        return np.array(img)


def predecode(tensor_layout: TensorLayout):
    # TODO
    # if jpeg, do ...
    # if h264, do ...

    # TODO
    return

    # default?
    # .reshape((-1, *input_shape))


def from_buffer(buf: ByteString, shape: tuple, dtype: type) -> np.ndarray:
    return np.frombuffer(buf, dtype=dtype).reshape(shape)


# TODO rename to "_decode_array" or "_decode_raw"? In contrast to JPEG and h264
# def _decode_data(model: keras.Model, data: ByteString) -> np.ndarray:
#     # Handle client side
#     if model is None:
#         # TODO np.float32? should probably store more info about model
#         return np.frombuffer(data, dtype=np.float32)[None, ...]
#     input_shape = model.layers[1].input_shape[1:]
#     input_type = model.layers[0].dtype
#     dtype = to_np_dtype(input_type)
#     return np.frombuffer(data, dtype=dtype).reshape((-1, *input_shape))


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


# TODO
# class H264Predecoder helps maintain state
# predecoder :: jpeg/h264 (ByteString -> np.ndarray [tiled])
# where default pre-decoder just returns array by directly decoding...
# detile :: (np.ndarray [tiled] -> np.ndarray [tensor])
# return
# format can be determined by input shape to server-side model
# or perhaps have server send params? idk
