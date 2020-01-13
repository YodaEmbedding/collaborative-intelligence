#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import time
import traceback
from asyncio import StreamReader, StreamWriter
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import ByteString, Dict

import numpy as np

from src.layouts import Layout, RgbLayout, TensorLayout, TiledArrayLayout
from src.modelconfig import ModelConfig, PostencoderConfig, ProcessorConfig
from src.predecode import (
    JpegPredecoder,
    JpegRgbPredecoder,
    Predecoder,
    RgbPredecoder,
    TensorPredecoder,
    from_buffer,
)
from src.server import monitor_client
from src.server.comm import (
    json_confirmation,
    json_ping,
    json_ready,
    json_result,
)
from src.server.model_manager import ModelManager
from src.server.monitor_client import MonitorStats, image_preview
from src.server.reader import read_item
from src.server.work_distributor import SmartProcessor, WorkDistributor
from src.tile import determine_tile_layout

IP = "0.0.0.0"
PORT = 5678
PORT2 = 5680


def str_preview(s: ByteString, max_len=16):
    if len(s) < max_len:
        return s.hex()
    return f"{s[:max_len - 6].hex()}...{s[-3:].hex()}"


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


@dataclass
class State:
    model_config: ModelConfig = None
    predecoder: Predecoder = None
    tensor_layout: TensorLayout = None


# TODO document that this happens on a single dedicated thread
# TODO turn this into a class or something
def processor(work_distributor: WorkDistributor, monitor_stats: MonitorStats):
    """Process work items received from work distributor."""
    model_manager = ModelManager()
    smart_processor = SmartProcessor(work_distributor)
    states: Dict[int, State] = defaultdict(State)

    while True:
        try:
            guid, item = smart_processor.get()
            request_type, item = item
            state = states[guid]

            if request_type == "terminate":
                work_distributor.put(guid, None)
            elif request_type == "acquire":
                model_config = item
                assert state.model_config is None
                model_manager.acquire(model_config)
                state.model_config = model_config
                # TODO have client provide the tiled_layout
                state.tensor_layout = model_manager.input_tensor_layout(
                    model_config
                )
            elif request_type == "release":
                model_config = item
                assert model_config == state.model_config
                model_manager.release(model_config)
                state.model_config = None
            elif request_type == "init_postencoder":
                postencoder_config = item
                state.predecoder = get_predecoder(
                    postencoder_config, state.model_config, state.tensor_layout
                )
            elif request_type == "predict":
                frame_number, buf = item
                model_config = state.model_config
                confirmation = json_confirmation(
                    frame_number=frame_number, num_bytes=len(buf)
                )
                confirmation = f"{confirmation}\n".encode("utf8")
                work_distributor.put(guid, confirmation)
                # TODO predecode_time separately from inference_time
                t0 = time.time()
                data_tensor = state.predecoder.run(buf)
                data_tensor = data_tensor[np.newaxis, ...]
                preds = model_manager.predict(model_config, data_tensor)
                preds = model_manager.decode_predictions(model_config, preds)
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
            elif request_type == "ready":
                ready = json_ready(model_config=state.model_config)
                ready = f"{ready}\n".encode("utf8")
                work_distributor.put(guid, ready)
            elif request_type == "ping":
                id_ = item
                response = f"{json_ping(id_)}\n".encode("utf8")
                work_distributor.put(guid, response)
            else:
                raise ValueError("Unknown request type")
        except Exception:
            traceback.print_exc()


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
            # TODO merge with processor()?
            if input_type == "frame":
                frame_number, buf = item
                print(f"Produce: {frame_number} {str_preview(buf)}")
                with open("frame.dat", "wb") as f:
                    f.write(buf)
                await putter(("predict", item))
            # TODO why are all json input types handled in this way?
            elif input_type == "json":
                # TODO this is all very confusing... clarify why next_model_config exists and why we need prev_model_loaded
                print(f"Produce: {item}")
                processor_config = ProcessorConfig.from_json_dict(item)
                prev_model_config = model_config
                model_config = processor_config.model_config
                postencoder_config = processor_config.postencoder_config
                prev_valid = prev_model_config is not None
                changed = prev_valid and prev_model_config != model_config
                if changed:
                    await putter(("release", prev_model_config))
                if changed or not prev_valid:
                    await putter(("acquire", model_config))
                await putter(("init_postencoder", postencoder_config))
                await putter(("ready", None))
            elif input_type == "ping":
                await putter(("ping", item))
    finally:
        if model_config is not None:
            await putter(("release", model_config))
        await putter(("terminate", None))


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
            print("Write begin")
            writer.write(item)
            print("Drain...")
            await writer.drain()
            print("Write end")
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


# TODO read, inference, write in parallel, no? (multiprocess.executorpool)
