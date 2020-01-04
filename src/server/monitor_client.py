import asyncio
import base64
import json
import math
from asyncio import StreamReader, StreamWriter
from io import BytesIO
from typing import Awaitable, ByteString

import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

from src.tile import (
    TensorLayout,
    TiledArrayLayout,
    determine_tile_layout,
    tile,
)


class MonitorStats:
    def __init__(self):
        self.add(-1, -1, [], b"")

    def add(self, frame_number, inference_time, predictions, data):
        self.frame_number = frame_number
        self.inference_time = inference_time
        self.predictions = predictions
        self.data = data

    def json_dict(self) -> dict:
        return {
            "frameNumber": self.frame_number,
            "inferenceTime": self.inference_time,
            "predictions": self.predictions,
            "data": self.data,
        }


async def read_json(reader: StreamReader) -> Awaitable[dict]:
    return json.loads(await reader.readline())


def handle_client(monitor_stats: MonitorStats):
    async def client_handler(reader: StreamReader, writer: StreamWriter):
        print("New monitor client...")
        ip, port = writer.get_extra_info("peername")
        print(f"Connected to {ip}:{port}")

        while True:
            # request = await read_json(reader)
            d = monitor_stats.json_dict()
            response = json.dumps(d)
            writer.write(f"{response}\n".encode("utf8"))
            print(f"Monitor upload: {len(response)} B; {response[:50]}")
            print()
            # print(f"Monitor upload: {len(response) // 1000} KB")
            await writer.drain()
            await asyncio.sleep(0.2)

    return client_handler


def image_preview(data_tensor: np.ndarray) -> ByteString:
    def denorm(x):
        return (x * 255.99).astype(dtype=np.uint8)

    def colormap(x):
        cmap = cm.viridis
        rgba = cmap(x)
        return denorm(rgba[:, :, :3])

    # Handle softmax layer
    if len(data_tensor.shape) <= 2:
        return _b64png_encode(denorm(_squarify_1d(data_tensor)))

    # Handle grayscale image case
    if len(data_tensor.shape) <= 3:
        return _b64png_encode(data_tensor[0])

    # Handle RGB image case
    if data_tensor.shape[-1] <= 3:
        return _b64png_encode(data_tensor[0].astype(dtype=np.uint8))

    # Handle non-uint8 types by clipping to min/max
    if data_tensor.dtype != np.uint8:
        a = np.min(data_tensor)
        b = np.max(data_tensor)
        arr = _tile_tensor((data_tensor - a) / (b - a))
        return _b64png_encode(colormap(arr))

    norm = Normalize(vmin=0, vmax=255)
    arr = norm(_tile_tensor(data_tensor))
    return _b64png_encode(colormap(arr))


def _tile_tensor(data_tensor: np.ndarray) -> np.ndarray:
    data_tensor = data_tensor[0]
    h, w, c = data_tensor.shape
    tensor_layout = TensorLayout(c, h, w, "hwc")
    tiled_layout = determine_tile_layout(tensor_layout)
    return tile(data_tensor, tensor_layout, tiled_layout)


def _squarify_1d(data_tensor: np.ndarray) -> np.ndarray:
    c = data_tensor.size
    nrows = int(math.sqrt(c))
    ncols = math.ceil(c / nrows)
    t = data_tensor.reshape(-1).copy()
    t.resize(nrows * ncols)
    return t.reshape((nrows, ncols))


def _b64png_encode(arr: np.ndarray) -> ByteString:
    img = Image.fromarray(arr)
    with BytesIO() as buffer:
        img.save(buffer, "png")
        raw = base64.b64encode(buffer.getvalue()).decode("utf8")
        return f"data:image/png;base64,{raw}"
