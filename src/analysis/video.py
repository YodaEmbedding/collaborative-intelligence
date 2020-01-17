from typing import Iterator

import cv2
import numpy as np
import tensorflow.keras.backend as K  # pylint: disable=import-error

from src.layouts import TensorLayout, TiledArrayLayout
from src.tile import detile, tile


def read_video(
    filename: str, tensor_layout: TensorLayout, tiled_layout: TiledArrayLayout,
) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, :, 0]
        frame = detile(frame, tiled_layout, tensor_layout)
        yield frame

    cap.release()


def write_video(
    filename: str,
    frames: Iterator[np.ndarray],
    tensor_layout: TensorLayout,
    tiled_layout: TiledArrayLayout,
):
    height, width = tiled_layout.shape

    fourcc = cv2.VideoWriter_fourcc(*_fourcc_code(filename))
    writer = cv2.VideoWriter(
        filename, fourcc, fps=30, frameSize=(width, height), isColor=False
    )

    for frame in frames:
        frame = tile(frame, tensor_layout, tiled_layout)
        writer.write(frame)

    writer.release()


def _fourcc_code(filename: str):
    exts = {
        ".mp4": "mp4v",
        ".avi": "MJPG",
    }
    return next(v for k, v in exts if filename.endswith(k))
