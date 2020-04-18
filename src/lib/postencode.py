from io import BytesIO
from typing import ByteString, Callable

import numpy as np
from PIL import Image

from src.lib.layouts import TensorLayout
from src.lib.tile import determine_tile_layout, tile


class Postencoder:
    def run(self, arr: np.ndarray) -> ByteString:
        raise NotImplementedError


class CallablePostencoder(Postencoder):
    def __init__(self, func: Callable[[np.ndarray], ByteString]):
        self.func = func

    def run(self, arr: np.ndarray) -> ByteString:
        return self.func(arr)


class JpegPostencoder(Postencoder):
    MBU_SIZE = 16

    def __init__(self, tensor_layout: TensorLayout, **kwargs):
        self.kwargs = kwargs
        self.tensor_layout = tensor_layout
        self.tiled_layout = determine_tile_layout(tensor_layout)

    def run(self, arr: np.ndarray) -> ByteString:
        tiled_tensor = tile(arr, self.tensor_layout, self.tiled_layout)
        tiled_tensor = _pad(tiled_tensor, self.MBU_SIZE)
        client_bytes = _pil_encode(tiled_tensor, "JPEG", **self.kwargs)
        return client_bytes


class Jpeg2000Postencoder(Postencoder):
    MBU_SIZE = 16

    def __init__(
        self, tensor_layout: TensorLayout, out_size: int = None, **kwargs
    ):
        self.kwargs = kwargs
        self.tensor_layout = tensor_layout
        self.tiled_layout = determine_tile_layout(tensor_layout)
        if out_size is not None:
            size = np.prod(self.tiled_layout.shape)
            rate = size / out_size
            self.kwargs["quality_mode"] = "rates"
            self.kwargs["quality_layers"] = [rate]

    def run(self, arr: np.ndarray) -> ByteString:
        tiled_tensor = tile(arr, self.tensor_layout, self.tiled_layout)
        tiled_tensor = _pad(tiled_tensor, self.MBU_SIZE)
        client_bytes = _pil_encode(tiled_tensor, "JPEG2000", **self.kwargs)
        return client_bytes


class PngPostencoder(Postencoder):
    def __init__(self, tensor_layout: TensorLayout, **kwargs):
        self.kwargs = kwargs
        self.tensor_layout = tensor_layout
        self.tiled_layout = determine_tile_layout(tensor_layout)

    def run(self, arr: np.ndarray) -> ByteString:
        tiled_tensor = tile(arr, self.tensor_layout, self.tiled_layout)
        client_bytes = _pil_encode(tiled_tensor, "PNG", **self.kwargs)
        return client_bytes


class H264Postencoder(Postencoder):
    MBU_SIZE = 16

    def __init__(self, tensor_layout: TensorLayout):
        self.tensor_layout = tensor_layout
        self.tiled_layout = determine_tile_layout(tensor_layout)

    def run(self, arr: np.ndarray) -> ByteString:
        tiled_tensor = tile(arr, self.tensor_layout, self.tiled_layout)
        tiled_tensor = np.tile(tiled_tensor[..., np.newaxis], 3)
        tiled_tensor = _pad(tiled_tensor, self.MBU_SIZE)
        # client_bytes = _h264_encode(tiled_tensor, self.quality)
        # TODO
        # return client_bytes


def _pil_encode(
    arr: np.ndarray, img_format: str, optimize=True, **kwargs
) -> ByteString:
    img = Image.fromarray(arr)
    with BytesIO() as buf:
        img.save(buf, img_format, optimize=optimize, **kwargs)
        buf.seek(0)
        return buf.read()


def _pad(img: np.ndarray, mbu_size: int) -> np.ndarray:
    py = -img.shape[0] % mbu_size
    px = -img.shape[1] % mbu_size
    z = [(0, 0) for i in range(2, img.ndim)]
    return np.pad(img, ((0, py), (0, px), *z))
