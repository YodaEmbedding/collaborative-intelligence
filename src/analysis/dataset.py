import json
import urllib.request
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

data_dir = "data"


def single_sample_image() -> np.ndarray:
    return np.array(Image.open("sample224.jpg"))


def _get_imagenet_labels() -> Dict[str, List[str]]:
    try:
        with open("imagenet_class_index.json", "r") as f:
            json_data = f.read()
    except FileNotFoundError:
        url = (
            "https://storage.googleapis.com/download.tensorflow.org/data/"
            "imagenet_class_index.json"
        )
        response = urllib.request.urlopen(url)
        json_data = response.read()
        with open("imagenet_class_index.json", "wb") as f:
            f.write(json_data)

    return json.loads(json_data)


def _get_imagenet_reverse_lookup() -> Dict[str, int]:
    d = _get_imagenet_labels()
    return {name: int(idx) for idx, (name, label) in d.items()}


# TODO put this into a function that you can just @cache
imagenet_lookup = _get_imagenet_reverse_lookup()


def dataset(shuffle: bool = True) -> tf.data.Dataset:
    return dataset_kb(kb=30, filtered=False, shuffle=shuffle)


def dataset_kb(
    kb: int = 30, filtered: bool = True, shuffle: bool = True
) -> tf.data.Dataset:
    filename = "data_kb_filtered.csv" if filtered else "data_kb_all.csv"
    df = pd.read_csv(f"{data_dir}/{filename}")
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    filepaths = df["file"].map(lambda x: f"{data_dir}/{kb}kb/{x}")
    labels = df["label"].replace(imagenet_lookup).astype(dtype=np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(_parse_row)
    return dataset


def _parse_row(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw)
    img = tf.cast(img, tf.float32)
    return img, label
