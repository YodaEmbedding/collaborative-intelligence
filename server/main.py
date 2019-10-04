#!/usr/bin/env python3

import importlib.util
import itertools
import json
import os
import socket
import sys
import time
from contextlib import closing, suppress
from datetime import datetime
from typing import Any, ByteString, List, Optional, Tuple

import cv2
from keras.applications import imagenet_utils
import numpy as np
import tensorflow as tf
from tensorflow import keras

spec = importlib.util.spec_from_file_location(
    "model_def",
    "../tools/split.py",
)
model_def = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_def)

IP = "0.0.0.0"
PORT = 5678


def read_eol(conn):
    return conn.recv(1, socket.MSG_WAITALL) == b"\x00"


def read_fixed_message(conn) -> Optional[ByteString]:
    msg_len_buf = conn.recv(4, socket.MSG_WAITALL)
    if len(msg_len_buf) != 4 or not read_eol(conn):
        return None
    msg_len = int.from_bytes(msg_len_buf, byteorder="big")
    buf = conn.recv(msg_len, socket.MSG_WAITALL)
    if len(buf) < msg_len or not read_eol(conn):
        return None
    return buf


def read_image(conn):
    buf = read_fixed_message(conn)
    if buf is None:
        return None
    buf = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


def read_tensor_frame(conn) -> Optional[Tuple[int, ByteString]]:
    frame_number_buf = conn.recv(4, socket.MSG_WAITALL)
    if len(frame_number_buf) != 4 or not read_eol(conn):
        return None
    frame_number = int.from_bytes(frame_number_buf, byteorder="big")
    msg = read_fixed_message(conn)
    if msg is None:
        return None
    return frame_number, msg


def str_preview(s: ByteString, max_len=16):
    if len(s) < max_len:
        return s.hex()
    return f"{s[:max_len - 6].hex()}...{s[-3:].hex()}"


def tf_to_np_dtype(tf_dtype):
    return {tf.uint8: np.uint8, tf.float32: np.float32}[tf_dtype]


def decode_data(model, data, dtype=tf.float32):
    input_shape = model.layers[1].input_shape[1:]
    t = np.frombuffer(data, dtype=tf_to_np_dtype(dtype)).reshape(
        (-1, *input_shape)
    )
    return t


def decode_predictions(predictions):
    decoded_pred = imagenet_utils.decode_predictions(predictions)[0][:3]
    return [(name, desc, float(score)) for name, desc, score in decoded_pred]


def json_result(
    frame_number: int,
    read_time: int,
    feed_time: int,
    inference_time: int,
    predictions: List[List[Any]],
) -> str:
    return json.dumps(
        {
            "frameNumber": frame_number,
            "readTime": read_time,
            "feedTime": feed_time,
            "inferenceTime": inference_time,
            "predictions": [
                {"name": name, "description": desc, "score": score}
                for name, desc, score in predictions
            ],
        }
    )


class SockMonkey:
    """Simulate a socket that generates connections."""

    def __init__(self, filename):
        self._filename = filename

    def close(self):
        pass

    def accept(self):
        return ConnMonkey(self._filename), ("Virtual Address", -1)


class ConnMonkey:
    """Simulate a connection using given file data cyclically."""

    FPS = 0.0

    def __init__(self, filename):
        self._pos = 0
        with open(filename, "rb") as f:
            self._data = f.read()
        self._data = (
            len(self._data).to_bytes(4, byteorder="big")
            + b"\x00"
            + self._data
            + b"\x00"
        )

    def close(self):
        pass

    def recv(self, num_bytes, flags):
        if flags != socket.MSG_WAITALL:
            raise ValueError("Unsupported flags")
        xs = []
        remaining = num_bytes
        while remaining > 0:
            x = self._data[self._pos : self._pos + remaining]
            self._pos += len(x)
            time.sleep(self._pos // len(self._data) * ConnMonkey.FPS)
            self._pos %= len(self._data)
            remaining -= len(x)
            xs.append(x)
        result = b"".join(xs)
        return result

    def send(self, s: ByteString):
        print(s.decode("utf8"))


class Main:
    def __init__(self, debug, dtype, model_name):
        self.dtype = dtype

        model_filename = f"../tools/resnet34/resnet34-add_8-UniformQuantizationU8Encoder(clip_range=(-1.0, 1.0))-UniformQuantizationU8Decoder(clip_range=(-1.0, 1.0))-server.h5"

        print("Loading model...")
        self.sess = tf.Session()
        self.model = keras.models.load_model(
            model_filename,
            custom_objects={
                "UniformQuantizationU8Decoder": model_def.UniformQuantizationU8Decoder
            },
        )

        if not debug:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((IP, PORT))
            self.sock.listen(1)
        else:
            self.sock = SockMonkey("resnet34-float-add_8-monitor.dat")

    def start_connection(self):
        print("Waiting for connection...")
        conn, addr = self.sock.accept()
        print(f"Established connection on\n{conn}\n{addr}")

        data, data_tensor = None, None
        for i in itertools.count():
            result = self._loop_once(i, conn)
            if result is None:
                break
            data, data_tensor = result

        print("Closing connection...")
        conn.close()

        if data is None or data_tensor is None:
            return

        with suppress(NameError):
            with open("last_run_final_frame.dat", "wb") as f:
                f.write(data)
            np.save("last_run_final_frame.npy", data_tensor)

    def _loop_once(self, i, conn):
        t0 = time.time()
        msg = read_tensor_frame(conn)
        if msg is None:
            return None
        frame_number, data = msg

        now = datetime.now()

        t1 = time.time()
        data_tensor = decode_data(self.model, data, dtype=self.dtype)

        t2 = time.time()
        predictions = self.model.predict(data_tensor)

        t3 = time.time()
        decoded_pred = decode_predictions(predictions)
        decoded_pred_str = "\n".join(
            f"{name:12} {desc:24} {score:0.3f}"
            for name, desc, score in decoded_pred
        )
        msg = json_result(
            frame_number=frame_number,
            read_time=int(1000 * (t1 - t0)),
            feed_time=int(1000 * (t2 - t1)),
            inference_time=int(1000 * (t3 - t2)),
            predictions=decoded_pred,
        )
        conn.send(f"{msg}\n".encode("utf8"))

        t4 = time.time()

        print(now.isoformat(sep=" ", timespec="milliseconds"))
        print(i, len(data), str_preview(data))
        print(decoded_pred_str)
        print(f"Read:       {1000 * (t1 - t0):4.0f} ms")
        print(f"Feed input: {1000 * (t2 - t1):4.0f} ms")
        print(f"Inference:  {1000 * (t3 - t2):4.0f} ms")
        print(f"Send:       {1000 * (t4 - t3):4.0f} ms")
        print(f"Total:      {1000 * (t4 - t0):4.0f} ms")
        print("")

        return data, data_tensor


def main():
    DEBUG = False
    DTYPE = tf.uint8
    MODEL_NAME = "resnet34"
    # MODEL_NAME = "vgg19-block4_pool"
    # MODEL_NAME = "mobilenet_v1_1.0_224"

    main = Main(DEBUG, DTYPE, MODEL_NAME)

    with closing(main.sock):
        while True:
            try:
                main.start_connection()
            except ConnectionResetError as e:
                print(e)
            except BrokenPipeError as e:
                print(e)


if __name__ == "__main__":
    main()

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
