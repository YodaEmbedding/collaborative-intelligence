#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import base64
import json
import math
import queue
import time
import traceback
from asyncio import StreamReader, StreamWriter
from io import BytesIO
from itertools import count
from typing import (
    Any,
    Awaitable,
    ByteString,
    Dict,
    Generator,
    Generic,
    List,
    Tuple,
    TypeVar,
)

import janus
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import cm
from matplotlib.colors import Normalize
from tensorflow import keras

from src import monitor_client
from src.server.model_manager import ModelManager
from src.modelconfig import ModelConfig
from src.monitor_client import MonitorStats
from src.tile import (
    tile,
    determine_tile_layout,
    TensorLayout,
    TiledArrayLayout,
)

IP = "0.0.0.0"
PORT = 5678
PORT2 = 5680


def str_preview(s: ByteString, max_len=16):
    if len(s) < max_len:
        return s.hex()
    return f"{s[:max_len - 6].hex()}...{s[-3:].hex()}"


def tile_tensor(data_tensor: np.ndarray) -> np.ndarray:
    data_tensor = data_tensor[0]
    h, w, c = data_tensor.shape
    tensor_layout = TensorLayout(c, h, w, "hwc")
    tiled_layout = determine_tile_layout(tensor_layout)
    return tile(data_tensor, tensor_layout, tiled_layout)


def squarify_1d(data_tensor: np.ndarray) -> np.ndarray:
    c = data_tensor.size
    nrows = int(math.sqrt(c))
    ncols = math.ceil(c / nrows)
    t = data_tensor.reshape(-1).copy()
    t.resize(nrows * ncols)
    return t.reshape((nrows, ncols))


def b64png_encode(arr: np.ndarray) -> ByteString:
    img = Image.fromarray(arr)
    with BytesIO() as buffer:
        img.save(buffer, "png")
        raw = base64.b64encode(buffer.getvalue()).decode("utf8")
        return f"data:image/png;base64,{raw}"


def image_preview(data_tensor: np.ndarray) -> ByteString:
    def denorm(x):
        return (x * 255.99).astype(dtype=np.uint8)

    def colormap(x):
        cmap = cm.viridis
        rgba = cmap(x)
        return denorm(rgba[:, :, :3])

    # Handle softmax layer
    if len(data_tensor.shape) <= 2:
        return b64png_encode(denorm(squarify_1d(data_tensor)))

    # Handle grayscale image case
    if len(data_tensor.shape) <= 3:
        return b64png_encode(data_tensor[0])

    # Handle RGB image case
    if data_tensor.shape[-1] <= 3:
        return b64png_encode(data_tensor[0].astype(dtype=np.uint8))

    # Handle non-uint8 types by clipping to min/max
    if data_tensor.dtype != np.uint8:
        a = np.min(data_tensor)
        b = np.max(data_tensor)
        arr = tile_tensor((data_tensor - a) / (b - a))
        return b64png_encode(colormap(arr))

    norm = Normalize(vmin=0, vmax=255)
    arr = norm(tile_tensor(data_tensor))
    return b64png_encode(colormap(arr))


def json_result(
    frame_number: int,
    # read_time: int,
    # feed_time: int,
    inference_time: int,
    predictions: List[Tuple[str, str, float]],
) -> str:
    return json.dumps(
        {
            "type": "result",
            "frameNumber": frame_number,
            # "readTime": read_time,  # TODO?
            # "feedTime": feed_time,
            "inferenceTime": inference_time,
            "predictions": [
                {"name": name, "description": desc, "score": score}
                for name, desc, score in predictions
            ],
        }
    )


def json_confirmation(frame_number: int, num_bytes: int) -> str:
    return json.dumps(
        {
            "type": "confirmation",
            "frameNumber": frame_number,
            "numBytes": num_bytes,
        }
    )


def json_ready(model_config: ModelConfig) -> str:
    return json.dumps(
        {"type": "ready", "modelConfig": model_config.to_json_object()}
    )


def json_ping(id_) -> str:
    return json.dumps({"type": "ping", "id": id_})


R = TypeVar("R")
T = TypeVar("T")


class WorkDistributor(Generic[T, R]):
    """Process async items synchronously.

    Queues asynchronous requests for synchronous processing. Once
    processor is ready, it reads item from request queue, then puts the
    result into the result queue.
    """

    _requests: janus.Queue
    _results: Dict[int, janus.Queue]

    def __init__(self):
        self._guid = 0
        self._requests = janus.Queue()
        self._results = {}

    def register(self):
        """Register client for processing.

        Returns:
            request_callback: Asynchronously push request.
            result_callback: Asynchronously receive result.
        """
        guid = self._guid
        self._guid += 1
        self._results[guid] = janus.Queue()

        async def put_request(item: T):
            await self._requests.async_q.put((guid, item))

        async def get_result() -> Awaitable[R]:
            return await self._results[guid].async_q.get()

        return put_request, get_result

    def get(self) -> Tuple[int, T]:
        """Synchronously retrieve request for processing."""
        return self._requests.sync_q.get()

    def get_many(
        self, min_items=1, max_items=None
    ) -> Generator[Tuple[int, T], None, None]:
        """Synchronously retrieve requests for processing.

        Retrieve at least min_items, blocking if necessary. Retreive up
        to max_items if possible without blocking.
        """
        for _ in range(min_items):
            yield self.get()
        it = count() if max_items is None else range(max_items - min_items)
        for _ in it:
            if self._requests.sync_q.empty():
                break
            yield self.get()

    def empty(self) -> bool:
        """Check if process queue is empty."""
        return self._requests.sync_q.empty()

    def put(self, guid: int, item: R):
        """Synchronously push processed result."""
        self._results[guid].sync_q.put(item)


class SmartProcessor(Generic[T, R]):
    """Looks ahead to determine if work items should be cancelled."""

    def __init__(self, work_distributor: WorkDistributor[T, R]):
        self.work_distributor = work_distributor
        self.buffer = queue.Queue()

    def get(self) -> Tuple[int, T]:
        self._refresh_buffer()
        return self.buffer.get()

    def _refresh_buffer(self):
        min_items = 1 if self.buffer.empty() else 0
        items = list(self.work_distributor.get_many(min_items=min_items))
        idxs = (
            len(items) - i - 1
            for i, (_, (_, request_type, _)) in enumerate(reversed(items))
            if request_type == "release"
        )
        idx = next(idxs, None)
        if idx is None:
            for x in items:
                self.buffer.put(x)
            return
        self.buffer = queue.Queue()
        for x in items[idx:]:
            self.buffer.put(x)


# TODO Propogate exceptions back to client?
def processor(work_distributor: WorkDistributor, monitor_stats: MonitorStats):
    """Process work items received from work distributor."""
    model_manager = ModelManager()
    smart_processor = SmartProcessor(work_distributor)

    while True:
        try:
            guid, item = smart_processor.get()
            model_config, request_type, item = item

            if request_type == "terminate":
                work_distributor.put(guid, None)
            elif request_type == "acquire":
                model_manager.acquire(model_config)
                ready = json_ready(model_config=model_config)
                ready = f"{ready}\n".encode("utf8")
                work_distributor.put(guid, ready)
            elif request_type == "release":
                model_manager.release(model_config)
            elif request_type == "predict":
                frame_number, data = item
                confirmation = json_confirmation(
                    frame_number=frame_number, num_bytes=len(data)
                )
                confirmation = f"{confirmation}\n".encode("utf8")
                work_distributor.put(guid, confirmation)
                # TODO decode here, especially for MP4s...
                t0 = time.time()
                data_tensor = model_manager.decode_data(model_config, data)
                preds = model_manager.predict(model_config, data_tensor)
                t1 = time.time()
                inference_time = int(1000 * (t1 - t0))
                result = json_result(
                    frame_number=frame_number,
                    inference_time=int(1000 * (t1 - t0)),
                    predictions=preds,
                )
                result = f"{result}\n".encode("utf8")
                work_distributor.put(guid, result)
                monitor_stats.add(
                    frame_number=frame_number,
                    # data_shape=..., # TODO different shapes for data?
                    inference_time=inference_time,
                    predictions=preds,
                    data=image_preview(data_tensor),
                )
            elif request_type == "ping":
                id_ = item
                response = f"{json_ping(id_)}\n".encode("utf8")
                work_distributor.put(guid, response)
        except Exception:
            traceback.print_exc()


# TODO honestly, the "eol"s are a bit pointless...
async def read_eol(reader: StreamReader) -> Awaitable[bool]:
    return await reader.readexactly(1) == b"\x00"


async def read_int(reader: StreamReader) -> Awaitable[int]:
    return int.from_bytes(await reader.readexactly(4), byteorder="big")


async def read_tensor_frame(
    reader: StreamReader,
) -> Awaitable[Tuple[int, ByteString]]:
    frame_number = await read_int(reader)
    # print(frame_number)
    # await read_eol(reader)
    data_len = await read_int(reader)
    # print(data_len)
    # await read_eol(reader)
    data = await reader.readexactly(data_len)
    # print(str_preview(data))
    # await read_eol(reader)
    return frame_number, data


async def read_json(reader: StreamReader) -> Awaitable[dict]:
    return json.loads(await reader.readline())


async def read_ping(reader: StreamReader) -> Awaitable:
    id_ = await read_int(reader)
    return id_


# TODO form protocol on json and binary data
async def read_item(reader: StreamReader) -> Awaitable[Tuple[str, Any]]:
    """Retrieve single item of various types from stream."""
    input_type = (await reader.readline()).decode("utf8").rstrip("\n")
    print(input_type)
    if len(input_type) == 0:
        return "terminate", None
    reader_func = {
        "frame": read_tensor_frame,
        "json": read_json,
        "ping": read_ping,
    }[input_type]
    return input_type, await reader_func(reader)


async def produce(reader: StreamReader, putter):
    """Reads from socket, and pushes requests to processor."""
    model_config: ModelConfig = None

    try:
        while True:
            print("Read begin")
            input_type, item = await read_item(reader)
            print("Read end")
            if input_type == "terminate":
                break
            # TODO instead of always passing model_config, why not make class?
            # TODO merge with processor()?
            if input_type == "frame":
                i, data = item
                print(f"Produce: {i} {str_preview(data)}")
                await putter((model_config, "predict", item))
            # TODO why are all json input types handled in this way?
            elif input_type == "json":
                item = {k: v for k, v in item.items() if v is not None}
                print(f"Produce: {item}")
                next_model_config = ModelConfig(**item)
                valid = model_config is not None
                same = valid and model_config == next_model_config
                if same:
                    continue
                if valid:
                    await putter((model_config, "release", None))
                model_config = next_model_config
                await putter((model_config, "acquire", None))
            elif input_type == "ping":
                await putter((model_config, "ping", item))
    finally:
        if model_config is not None:
            await putter((model_config, "release", None))
        await putter((None, "terminate", None))


async def consume(writer: StreamWriter, getter):
    """Receives items and writes them to socket."""
    try:
        for i in count():
            item = await getter()
            if item is None:
                break
            item_d = json.loads(item.decode("utf8"))
            item_d.pop("predictions", None)
            print(f"Consume {i}: {item_d}")
            # print(json.dumps(item_d, indent=4))
            # TODO is this correct? await drain?
            print("Write begin")
            writer.write(item)
            print("Drain...")
            await writer.drain()
            print("Write end")
    # TODO what does close even do? do we need the finally?
    finally:
        print("Closing client...")
        writer.close()


def handle_client(work_distributor: WorkDistributor):
    async def client_handler(reader: StreamReader, writer: StreamWriter):
        print("New client...")
        ip, port = writer.get_extra_info("peername")
        print(f"Connected to {ip}:{port}")
        putter, getter = work_distributor.register()
        coros = [produce(reader, putter), consume(writer, getter)]
        tasks = map(asyncio.create_task, coros)
        await asyncio.wait(tasks)

    return client_handler


async def main():
    work_distributor = WorkDistributor()
    monitor_stats = MonitorStats()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, processor, work_distributor, monitor_stats)
    client_handler = handle_client(work_distributor)
    server = await asyncio.start_server(client_handler, IP, PORT)
    monitor_handler = monitor_client.handle_client(monitor_stats)
    monitor_server = await asyncio.start_server(monitor_handler, IP, PORT2)
    print("Started server")
    await asyncio.wait(
        [server.serve_forever(), monitor_server.serve_forever()]
    )


if __name__ == "__main__":
    asyncio.run(main())


# TODO
# HTTP binary data
# How do we want to design this? One model, multithread? Model manager? Etc?
# RxPy?
# Switch models by JSON (client tells us what model to serve)
# { "model": "resnet34", "splitLayer": "add_8", "encoder": "uniquant", "decoder": "uniquant", "encoderArgs": [-2, 2], "decoderArgs": [-2, 2] }
# For now, single requests/model...
# print statements...? or update a TUI?
# Multiple clients, scarce resources (GPU)
# Deal with overload of requests from clients (skip outdated requests)
# More resources: load multiple instances of model
# Handle client reconnection attempts gracefully
# Signal loss from client -> timeout on recv
# Exception handling: https://docs.python.org/3/library/socket.html#example


# TODO how to deal with reconfigure_request received at same time as inference_request? or is that a problem only if using more than one thread... but how to deal with starvation then? HMMMM or backpressure/buffers
# TODO try with time.sleep()
# TODO read, inference, write in parallel, no? (multiprocess.executorpool)
# TODO switch to asyncio based server? Or with StreamReader/StreamWriter
# TODO finally on drop connection: release()
# NOTE single threaded inference is probably better to prevent starvation anyways...!
# TODO batch scheduling?
# TODO Tensorflow Serving? run on localhost if want additional functionality (like stats)
