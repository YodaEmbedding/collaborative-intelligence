#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import time
import traceback
from asyncio import StreamReader, StreamWriter
from itertools import count
from typing import Any, Awaitable, ByteString, List, Tuple

from src.modelconfig import ModelConfig
from src.server import monitor_client
from src.server.model_manager import ModelManager
from src.server.monitor_client import MonitorStats, image_preview
from src.server.work_distributor import SmartProcessor, WorkDistributor

IP = "0.0.0.0"
PORT = 5678
PORT2 = 5680


def str_preview(s: ByteString, max_len=16):
    if len(s) < max_len:
        return s.hex()
    return f"{s[:max_len - 6].hex()}...{s[-3:].hex()}"


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


async def read_int(reader: StreamReader) -> Awaitable[int]:
    return int.from_bytes(await reader.readexactly(4), byteorder="big")


async def read_tensor_frame(
    reader: StreamReader,
) -> Awaitable[Tuple[int, ByteString]]:
    frame_number = await read_int(reader)
    data_len = await read_int(reader)
    data = await reader.readexactly(data_len)
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
