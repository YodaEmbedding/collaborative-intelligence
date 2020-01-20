from io import BytesIO
from typing import ByteString

import numpy as np
from PIL import Image

from src.layouts import TensorLayout
from src.modelconfig import PostencoderConfig
from src.tile import determine_tile_layout, tile


class Postencoder:
    def run(self, arr: np.ndarray) -> ByteString:
        raise NotImplementedError


class JpegPostencoder(Postencoder):
    MBU_SIZE = 16

    def __init__(
        self,
        tensor_layout: TensorLayout,
        postencoder_config: PostencoderConfig,
    ):
        self.tensor_layout = tensor_layout
        self.postencoder_config = postencoder_config

    def run(self, arr: np.ndarray) -> ByteString:
        tensor_layout = self.tensor_layout
        quality = self.postencoder_config.quality
        tiled_layout = determine_tile_layout(tensor_layout)
        tiled_tensor = tile(arr, tensor_layout, tiled_layout)
        tiled_tensor = np.tile(tiled_tensor[..., np.newaxis], 3)
        tiled_tensor = _pad(tiled_tensor, JpegPostencoder.MBU_SIZE)
        client_bytes = _jpeg_encode(tiled_tensor, quality)
        return client_bytes


def _jpeg_encode(arr: np.ndarray, quality: int) -> ByteString:
    img = Image.fromarray(arr)
    with BytesIO() as buf:
        img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return buf.read()


def _pad(img: np.ndarray, mbu_size: int) -> np.ndarray:
    py = -img.shape[0] % mbu_size
    px = -img.shape[1] % mbu_size
    return np.pad(img, ((0, py), (0, px), (0, 0)))
