import asyncio
import json
from asyncio import StreamReader, StreamWriter
from typing import Awaitable


# TODO synchronize add()/json_dict()? or actually, we're using async anyways...
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
