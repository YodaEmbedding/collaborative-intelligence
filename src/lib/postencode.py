from io import BytesIO
from typing import ByteString

import numpy as np
from PIL import Image

from src.lib.layouts import TensorLayout
from src.lib.tile import determine_tile_layout, tile


class Postencoder:
    def run(self, arr: np.ndarray) -> ByteString:
        raise NotImplementedError


class JpegPostencoder(Postencoder):
    MBU_SIZE = 16

    def __init__(
        self, tensor_layout: TensorLayout, quality: int = None,
    ):
        self.tensor_layout = tensor_layout
        self.quality = quality
        self.tiled_layout = determine_tile_layout(tensor_layout)

    def run(self, arr: np.ndarray) -> ByteString:
        tiled_tensor = tile(arr, self.tensor_layout, self.tiled_layout)
        tiled_tensor = _pad(tiled_tensor, JpegPostencoder.MBU_SIZE)
        client_bytes = _jpeg_encode(tiled_tensor, self.quality)
        return client_bytes


class H264Postencoder(Postencoder):
    def __init__(self, tensor_layout: TensorLayout):
        self.tensor_layout = tensor_layout
        self.tiled_layout = determine_tile_layout(tensor_layout)

    def run(self, arr: np.ndarray) -> ByteString:
        tiled_tensor = tile(arr, self.tensor_layout, self.tiled_layout)
        tiled_tensor = np.tile(tiled_tensor[..., np.newaxis], 3)
        tiled_tensor = _pad(tiled_tensor, JpegPostencoder.MBU_SIZE)
        # client_bytes = _h264_encode(tiled_tensor, self.quality)
        # TODO
        # return client_bytes


def _jpeg_encode(arr: np.ndarray, quality: int) -> ByteString:
    img = Image.fromarray(arr)
    with BytesIO() as buf:
        img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return buf.read()


def _pad(img: np.ndarray, mbu_size: int) -> np.ndarray:
    py = -img.shape[0] % mbu_size
    px = -img.shape[1] % mbu_size
    z = [(0, 0) for i in range(2, img.ndim)]
    return np.pad(img, ((0, py), (0, px), *z))
