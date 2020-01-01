import json
from asyncio import StreamReader
from typing import Any, Awaitable, ByteString, Tuple


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
