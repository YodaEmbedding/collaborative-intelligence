#!/usr/bin/env python3

import itertools
import socket
import time
from pprint import pprint
from typing import Dict, ByteString

import cv2
from keras.applications import imagenet_utils
import numpy as np
import tensorflow as tf
from tensorflow import keras

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
    if len(s) < max_len:
        return s
    return f'{s[:max_len - 6]}...{s[-3:]}'

def predict(sess, model, data):
    t = tf.decode_raw(input_bytes=data, out_type=float, little_endian=True)
    t = tf.reshape(t, (-1, 7, 7, 512))
    with sess.as_default():
        t = t.eval()
    return model.predict(t)

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

class SockMonkey:
    def close(self):
        pass

def main():
    DEBUG = False

    print('Loading model...')
    sess = tf.Session()
    model_name = 'mobilenet_v1_1.0_224'
    model = keras.models.load_model(f'{model_name}-server.h5')

    if not DEBUG:
        print('Waiting for connection...')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((IP, PORT))
        sock.listen(1)
        conn, addr = sock.accept()
        print(f'Established connection on\n{conn}\n{addr}')
    else:
        conn = ConnMonkey('monitor.dat')
        sock = SockMonkey()

    for i in itertools.count():
        data = read_fixed_message(conn)
        if data is None:
            break

        print(i, len(data), str_preview(data))
        predictions = predict(sess, model, data)
        pprint(imagenet_utils.decode_predictions(predictions)[0])

    # for i in itertools.count():
    #     img = read_image(conn)
    #     if img is None:
    #         break
    #
    #     print(i, img.shape)
    #     cv2.imshow('Preview', img)
    #     key = cv2.waitKey(30) & 0xff
    #     if key == 27:
    #         break
    #
    # cv2.destroyAllWindows()

    sock.close()

if __name__ == '__main__':
    main()
