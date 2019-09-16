#!/usr/bin/env python3

import importlib.util
import itertools
import os
import socket
import sys
import time
from contextlib import suppress
from pprint import pprint
from typing import Dict, ByteString

import cv2
from keras.applications import imagenet_utils
import numpy as np
import tensorflow as tf
from tensorflow import keras

spec = importlib.util.spec_from_file_location(
    'model_def',
    os.path.expandvars(
        '$HOME/code/experiments/py/tensorflow/resnet-keras-split/main.py'))
model_def = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_def)

IP = '0.0.0.0'
PORT = 5678

def read_eol(conn):
    return conn.recv(1, socket.MSG_WAITALL) == b'\x00'

def read_fixed_message(conn) -> ByteString:
    msg_len_buf = conn.recv(4, socket.MSG_WAITALL)
    if len(msg_len_buf) != 4 or not read_eol(conn):
        return None
    msg_len = int.from_bytes(msg_len_buf, byteorder='big')
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

def str_preview(s, max_len=16):
    def unquoted(x: ByteString) -> str:
        return repr(x)[2:-1]

    if len(s) < max_len:
        return s
    return f'{unquoted(s[:max_len - 6])}...{unquoted(s[-3:])}'

def decode_data(sess, model, data, dtype=tf.float32):
    input_shape = model.layers[1].input_shape[1:]
    t = tf.decode_raw(input_bytes=data, out_type=dtype, little_endian=True)
    t = tf.reshape(t, (-1, *input_shape))
    with sess.as_default():
        t = t.eval()
    return t

class SockMonkey:
    """Simulate a socket that generates connections that use given file data."""

    def __init__(self, filename):
        self._filename = filename

    def close(self):
        pass

    def accept(self):
        return ConnMonkey(self._filename), ('Virtual Address', -1)

class ConnMonkey:
    """Simulate a connection using given file data cyclically."""

    FPS = 0.2

    def __init__(self, filename):
        self._pos = 0
        with open(filename, 'rb') as f:
            self._data = f.read()
        self._data = (
            len(self._data).to_bytes(4, byteorder='big') +
            b'\x00' +
            self._data +
            b'\x00')

    def recv(self, num_bytes, flags):
        if flags != socket.MSG_WAITALL:
            raise ValueError('Unsupported flags')
        xs = []
        remaining = num_bytes
        while remaining > 0:
            x = self._data[self._pos:self._pos+remaining]
            self._pos += len(x)
            time.sleep(self._pos // len(self._data) * ConnMonkey.FPS)
            self._pos %= len(self._data)
            remaining -= len(x)
            xs.append(x)
        result = b''.join(xs)
        return result

def main():
    DEBUG = False
    DTYPE = tf.uint8
    model_name = 'resnet34'
    # model_name = 'mobilenet_v1_1.0_224'

    opt_suffix = '-float' if DTYPE == tf.float32 else ''
    model_filename = f'{model_name}-server{opt_suffix}.h5'

    print('Loading model...')
    sess = tf.Session()
    model = keras.models.load_model(
        model_filename,
        custom_objects={'DecoderLayer': model_def.DecoderLayer})

    if not DEBUG:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((IP, PORT))
        sock.listen(1)
    else:
        sock = SockMonkey('resnet34-float-add_8-monitor.dat')

    while True:
        print('Waiting for connection...')
        conn, addr = sock.accept()
        print(f'Established connection on\n{conn}\n{addr}')

        for i in itertools.count():
            t0 = time.time()
            msg = read_fixed_message(conn)
            if msg is None:
                break

            t1 = time.time()
            data = msg
            data_tensor = decode_data(sess, model, data, dtype=DTYPE)

            t2 = time.time()
            predictions = model.predict(data_tensor)

            t3 = time.time()
            decoded_pred = imagenet_utils.decode_predictions(predictions)[0]
            decoded_pred_str = '\n'.join(
                f'{name:12} {desc:24} {score:0.3f}'
                for name, desc, score in decoded_pred)

            print(i, len(data), str_preview(data))
            print(decoded_pred_str)
            print(f'Read:       {1000 * (t1 - t0):4.0f} ms')
            print(f'Feed input: {1000 * (t2 - t1):4.0f} ms')
            print(f'Inference:  {1000 * (t3 - t2):4.0f} ms')
            print(f'Total:      {1000 * (t3 - t0):4.0f} ms')
            print('')

        print('Closing connection...')
        conn.close()

        with suppress(NameError):
            with open('last_run_final_frame.dat', 'wb') as f:
                f.write(data)
            np.save('last_run_final_frame.npy', data_tensor)

    # NOTE This never executes...
    sock.close()

if __name__ == '__main__':
    main()

# TODO
# Multiple clients, scarce resources (GPU)
# Deal with overload of requests from clients (skip outdated requests)
# More resources: load multiple instances of model
# Handle client reconnection attempts gracefully
# Signal loss from client -> timeout on recv
# Exception handling: https://docs.python.org/3/library/socket.html#example
