#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import time
import traceback
from asyncio import StreamReader, StreamWriter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    ByteString,
    Dict,
    Generic,
    List,
    Tuple,
    TypeVar,
)

import janus
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils
from tensorflow import keras

spec = importlib.util.spec_from_file_location("layers", "../tools/layers.py")
layers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layers)
decoders = layers.decoders

spec = importlib.util.spec_from_file_location(
    "modelconfig", "../tools/modelconfig.py"
)
modelconfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modelconfig)
ModelConfig = modelconfig.ModelConfig

IP = "0.0.0.0"
PORT = 5678


def str_preview(s: ByteString, max_len=16):
    if len(s) < max_len:
        return s.hex()
    return f"{s[:max_len - 6].hex()}...{s[-3:].hex()}"


def to_np_dtype(dtype):
    return {
        "float32": np.float32,
        "uint8": np.uint8,
        tf.float32: np.float32,
        tf.uint8: np.uint8,
    }[dtype]


def decode_data(model: keras.Model, data: ByteString) -> np.ndarray:
    input_shape = model.layers[1].input_shape[1:]
    input_type = model.layers[0].dtype  # TODO verify
    t = np.frombuffer(data, dtype=to_np_dtype(input_type)).reshape(
        (-1, *input_shape)
    )
    return t


def json_result(
    frame_number: int,
    # read_time: int,
    # feed_time: int,
    inference_time: int,
    predictions: List[Tuple[str, str, float]],
) -> str:
    return json.dumps(
        {
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


@dataclass
class ModelReference:
    ref_count: int
    model: keras.Model


class ModelManager:
    def __init__(self):
        self.sess = tf.Session()
        self.models: Dict[ModelConfig, ModelReference] = {}

    def acquire(self, model_config: ModelConfig):
        if model_config not in self.models:
            print(f"Loaded model {model_config}")
            model = self._load_model(model_config)
            self.models[model_config] = ModelReference(0, model)
        self.models[model_config].ref_count += 1

    def release(self, model_config: ModelConfig):
        self.models[model_config].ref_count -= 1
        if self.models[model_config].ref_count == 0:
            print(f"Released model {model_config}")
            del self.models[model_config].model
            del self.models[model_config]

    # @synchronized
    def predict(
        self, model_config: ModelConfig, data: ByteString
    ) -> List[Tuple[str, str, float]]:
        model = self.models[model_config].model
        data_tensor = decode_data(model, data)
        predictions = model.predict(data_tensor)
        return self._decode_predictions(predictions)

    def _decode_predictions(
        self, predictions: np.ndarray, num_preds: int = 3
    ) -> List[Tuple[str, str, float]]:
        pred = imagenet_utils.decode_predictions(predictions)[0][:num_preds]
        return [(name, desc, float(score)) for name, desc, score in pred]

    def _load_model(self, model_config: ModelConfig) -> keras.Model:
        decoder = model_config.decoder
        custom_objects = {}
        if decoder != "None":
            custom_objects[decoder] = decoders[decoder]
        return keras.models.load_model(
            filepath=f"../tools/{model_config.to_path()}-server.h5",
            custom_objects=custom_objects,
        )


R = TypeVar("R")
T = TypeVar("T")


class WorkDistributor(Generic[T, R]):
    """Process async items synchronously."""

    _queue: janus.Queue
    _results: Dict[int, janus.Queue]

    def __init__(self):
        self._guid = 0
        self._queue = janus.Queue()
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
            await self._queue.async_q.put((guid, item))

        async def get_result() -> Awaitable[R]:
            return await self._results[guid].async_q.get()

        return put_request, get_result

    def get(self) -> Tuple[int, T]:
        """Synchronously retrieve item for processing."""
        return self._queue.sync_q.get()

    def empty(self) -> bool:
        """Check if process queue is empty."""
        return self._queue.sync_q.empty()

    def put(self, guid: int, item: R):
        """Synchronously push processed result."""
        self._results[guid].sync_q.put(item)


# TODO Propogate exceptions back to client?
def processor(work_distributor: WorkDistributor):
    model_manager = ModelManager()

    while True:
        try:
            guid, item = work_distributor.get()
            model_config, request_type, item = item

            if request_type == "terminate":
                work_distributor.put(guid, None)
            elif request_type == "acquire":
                model_manager.acquire(model_config)
            elif request_type == "release":
                model_manager.release(model_config)
            elif request_type == "predict":
                frame_number, data = item
                t0 = time.time()
                preds = model_manager.predict(model_config, data)
                t1 = time.time()
                result = json_result(
                    frame_number=frame_number,
                    inference_time=int(1000 * (t1 - t0)),
                    predictions=preds,
                )
                result = f"{result}\n".encode("utf8")
                work_distributor.put(guid, result)
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
    print(frame_number)
    # await read_eol(reader)
    data_len = await read_int(reader)
    print(data_len)
    # await read_eol(reader)
    data = await reader.readexactly(data_len)
    print(str_preview(data))
    # await read_eol(reader)
    return frame_number, data


async def read_json(reader: StreamReader) -> Awaitable[dict]:
    return json.loads(await reader.readline())


# TODO form protocol on json and binary data
async def read_item(reader: StreamReader) -> Awaitable[Tuple[str, Any]]:
    input_type = (await reader.readline()).decode("utf8").rstrip("\n")
    print(input_type)
    if len(input_type) == 0:
        return "terminate", None
    reader_func = {"frame": read_tensor_frame, "json": read_json}[input_type]
    return input_type, await reader_func(reader)


async def produce(reader: StreamReader, putter):
    model_config: ModelConfig = None

    # TODO DEBUG
    # TODO how to format floats/tuples?
    model_config = ModelConfig(
        "resnet34",
        "add_8",
        "UniformQuantizationU8Encoder",
        "UniformQuantizationU8Decoder",
        {"clip_range": [-1.0, 1.0]},
        {"clip_range": [-1.0, 1.0]},
    )
    await putter((model_config, "acquire", None))

    try:
        while True:
            input_type, item = await read_item(reader)
            print(
                f"Produce: {input_type}, {str_preview(str(item).encode('utf8'))}"
            )
            if input_type == "terminate":
                break
            # TODO instead of always passing model_config, why not make class?
            # TODO merge with processor()?
            elif input_type == "frame":
                await putter((model_config, "predict", item))
            elif input_type == "json":
                next_model_config = ModelConfig(**item)
                valid = model_config is not None
                same = valid and model_config == next_model_config
                if same:
                    continue
                if valid:
                    await putter((model_config, "release", None))
                model_config = next_model_config
                await putter((model_config, "acquire", None))
    finally:
        if model_config is not None:
            await putter((model_config, "release", None))
        await putter((None, "terminate", None))


async def consume(writer: StreamWriter, getter):
    try:
        while True:
            item = await getter()
            if item is None:
                break
            print("Consume:")
            print(json.dumps(json.loads(item.decode("utf8")), indent=4))
            writer.write(item)
            await writer.drain()
    # TODO what does close even do? do we need the finally?
    finally:
        print("Closing client...")
        writer.close()


def handle_client(work_distributor: WorkDistributor):
    async def client_handler(reader, writer):
        print("New client...")
        putter, getter = work_distributor.register()
        coros = [produce(reader, putter), consume(writer, getter)]
        tasks = map(asyncio.create_task, coros)
        await asyncio.wait(tasks)

    return client_handler


async def main():
    work_distributor = WorkDistributor()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, processor, work_distributor)
    client_handler = handle_client(work_distributor)
    server = await asyncio.start_server(client_handler, IP, PORT)
    await server.serve_forever()


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
