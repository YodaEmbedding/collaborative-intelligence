import textwrap
from collections import defaultdict
from os import path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from src.analysis import plot
from src.analysis import dataset as ds
from src.analysis.utils import basename_of, predict_dataset, title_of
from src.lib.layouts import TensorLayout
from src.lib.postencode import JpegPostencoder, Postencoder
from src.lib.predecode import JpegPredecoder, Predecoder

BYTES_PER_KB = 1000


# TODO loop postencoders...?
def analyze_accuracyvskb_layer(
    model_name: str,
    model: keras.Model,
    model_client: keras.Model,
    model_server: keras.Model,
    layer_name: str,
    layer_i: int,
    layer_n: int,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
    subdir: str = "",
):
    if len(model_client.output_shape) != 4:
        return

    title = title_of(model_name, layer_name, layer_i, layer_n)
    basename = basename_of(model_name, layer_name, layer_i, layer_n)
    basedir = "img/accuracyvskb"
    shareddir = path.join(basedir, subdir)
    filename_server = path.join(basedir, f"{model_name}-server.csv")
    filename_shared = path.join(shareddir, f"{basename}-shared.csv")

    try:
        data_server = pd.read_csv(filename_server)
    except FileNotFoundError:
        accuracies_server = _evaluate_accuracies_server_kb(model, batch_size)
        kbs_server = accuracies_server[0] / 1.024
        acc_server = accuracies_server[1]
        data_server = pd.DataFrame({"kbs": kbs_server, "acc": acc_server})
        data_server.to_csv(filename_server)
        print("Analyzed server accuracy vs KB")

    try:
        data_shared = pd.read_csv(filename_shared)
    except FileNotFoundError:
        accuracies_shared = _evaluate_accuracies_shared_kb(
            model_client, model_server, quant, dequant, batch_size
        )
        kbs_shared = accuracies_shared[0] / 1.024
        acc_shared = accuracies_shared[1]
        data_shared = pd.DataFrame({"kbs": kbs_shared, "acc": acc_shared})
        data_shared.to_csv(filename_shared)
        print("Analyzed shared accuracy vs KB")

    fig = _plot_accuracy_vs_kb(title, data_server, data_shared)
    plot.save(fig, path.join(shareddir, f"{basename}.png"))
    print("Analyzed accuracy vs KB")


def _compute_dataset_accuracies(
    model_client: keras.Model,
    model_server: keras.Model,
    postencoder: Postencoder,
    predecoder: Predecoder,
    dataset: tf.data.Dataset,
    accuracy_func: Callable[[np.ndarray, int], float],
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
) -> List[float]:
    accuracies = []

    for frames, labels in tfds.as_numpy(dataset):
        client_tensors = model_client.predict_on_batch(frames)
        decoded_tensors = []
        for client_tensor in client_tensors:
            quant_tensor = quant(client_tensor)
            encoded_bytes = postencoder.run(quant_tensor)
            decoded_tensor = predecoder.run(encoded_bytes)
            recv_tensor = dequant(decoded_tensor)
            decoded_tensors.append(recv_tensor)
        decoded_tensors = np.array(decoded_tensors)
        predictions = model_server.predict_on_batch(decoded_tensors)
        accuracies.extend(accuracy_func(labels, predictions))

    return accuracies


def _evaluate_accuracy_kb(
    model: keras.Model, kb: int, batch_size: int
) -> np.ndarray:
    dataset = ds.dataset_kb(kb)
    predictions = predict_dataset(model, dataset.batch(batch_size))
    labels = np.array(list(tfds.as_numpy(dataset.map(_second))))
    accuracies = _categorical_top1_accuracy(labels, predictions)
    kbs = np.ones_like(accuracies) * kb
    return np.vstack((kbs, accuracies))


def _evaluate_accuracies_server_kb(
    model: keras.Model, batch_size: int,
) -> np.ndarray:
    accuracies_server = [
        _evaluate_accuracy_kb(model, kb, batch_size) for kb in range(1, 31)
    ]
    return np.concatenate(accuracies_server, axis=1)


def _evaluate_accuracies_shared_kb(
    model_client: keras.Model,
    model_server: keras.Model,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> np.ndarray:
    accuracies_shared = defaultdict(list)
    dataset = ds.dataset()
    client_tensors = predict_dataset(model_client, dataset.batch(batch_size))
    client_tensors = quant(client_tensors)
    quality_lookup = _make_quality_lut(client_tensors)
    kb_lookup = [{q: kb for kb, q in d.items()} for d in quality_lookup]
    tensor_layout = TensorLayout.from_shape(
        client_tensors.shape[1:], "hwc", client_tensors.dtype
    )

    for quality in range(1, 100):
        keep = [quality in d for d in kb_lookup]
        keep_ds = tf.data.Dataset.from_tensor_slices(keep)

        dataset_quality = (
            tf.data.Dataset.zip((dataset, keep_ds)).filter(_second).map(_first)
        )

        postencoder = JpegPostencoder(tensor_layout, quality=quality)
        tiled_layout = postencoder.tiled_layout
        predecoder = JpegPredecoder(tiled_layout, tensor_layout)

        accuracies = _compute_dataset_accuracies(
            model_client,
            model_server,
            postencoder,
            predecoder,
            dataset_quality.batch(batch_size),
            _categorical_top1_accuracy,
            quant,
            dequant,
        )

        kbs = [d[quality] for d in kb_lookup if quality in d]
        for kb, acc in zip(kbs, accuracies):
            accuracies_shared[kb].append(acc)

    accuracies_shared = [
        np.vstack((np.ones(len(xs)) * kb, np.array(xs)))
        for kb, xs in accuracies_shared.items()
    ]

    return np.concatenate(accuracies_shared, axis=1)


def _make_quality_lut(
    client_tensors: List[np.ndarray],
) -> List[Dict[int, int]]:
    quality_lookup = []
    for client_tensor in client_tensors:
        tensor_layout = TensorLayout.from_tensor(client_tensor, "hwc")
        d = {}
        for quality in range(1, 101):
            postencoder = JpegPostencoder(tensor_layout, quality=quality)
            encoded_bytes = postencoder.run(client_tensor)
            kb = int(len(encoded_bytes) / BYTES_PER_KB)
            if kb not in d:
                d[kb] = quality
        quality_lookup.append(d)
    return quality_lookup


def _plot_accuracy_vs_kb(
    title: str, data_server: pd.DataFrame, data_shared: pd.DataFrame
):
    fig = plt.figure()
    ax = sns.lineplot(x="kbs", y="acc", data=data_server)
    ax = sns.lineplot(x="kbs", y="acc", data=data_shared)
    ax: plt.Axes = plt.gca()
    ax.legend(
        labels=["server-only inference", "shared inference"],
        loc="lower right",
    )
    ax.set(xlim=(0, 30), ylim=(0, 1))
    ax.set_xlabel("KB/frame")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize="xx-small")
    return fig


def _categorical_top1_accuracy(
    label: np.ndarray, pred: np.ndarray
) -> np.ndarray:
    return (np.argmax(pred, axis=-1) == label).astype(np.float32)


@tf.function(autograph=False)
def _first(x, _y):
    return x


@tf.function(autograph=False)
def _second(_x, y):
    return y
