import textwrap
from collections import defaultdict
from os import path
from typing import Callable, Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from src.analysis import dataset as ds
from src.analysis import plot
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
        data_server = _evaluate_accuracies_server_kb(model, batch_size)
        data_server.to_csv(filename_server, index=False)
        print("Analyzed server accuracy vs KB")

    try:
        data_shared = pd.read_csv(filename_shared)
    except FileNotFoundError:
        data_shared = _evaluate_accuracies_shared_kb(
            model_client, model_server, quant, dequant, batch_size
        )
        data_shared.to_csv(filename_shared, index=False)
        print("Analyzed shared accuracy vs KB")

    if "kbs" not in data_shared.columns:
        data_shared["kbs"] = data_shared["bytes"] / 1024
        data_shared.drop("bytes", axis=1, inplace=True)

    bins = np.logspace(0, np.log10(30), num=30)
    data_shared = _bin_for_plot(data_shared, "kbs", bins)

    fig = _plot_accuracy_vs_kb(title, data_server, data_shared)
    plot.save(fig, path.join(shareddir, f"{basename}.png"))
    print("Analyzed accuracy vs KB")


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
) -> pd.DataFrame:
    accuracies_server = [
        _evaluate_accuracy_kb(model, kb, batch_size) for kb in range(1, 31)
    ]
    accuracies_server = np.concatenate(accuracies_server, axis=1)
    kbs_server = accuracies_server[0] / 1.024
    acc_server = accuracies_server[1]
    return pd.DataFrame({"kbs": kbs_server, "acc": acc_server})


def _evaluate_accuracies_shared_kb(
    model_client: keras.Model,
    model_server: keras.Model,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> pd.DataFrame:
    dataset = ds.dataset()
    client_tensors = predict_dataset(model_client, dataset.batch(batch_size))
    labels = np.array(list(tfds.as_numpy(dataset.map(_second))))

    # TODO for JPEG2000, rate control rather than quality
    many_func = lambda x: _generate_jpeg_tensors(x, quant, dequant)
    batches = _make_batches(client_tensors, labels, many_func, batch_size)

    byte_sizes = []
    accuracies = []
    labels = []

    for xs, ls, bs in batches:
        xs = model_server.predict_on_batch(xs)
        accs = _categorical_top1_accuracy(ls, xs)
        accuracies.extend(accs)
        labels.extend(ls)
        byte_sizes.extend(bs)
        print(f"processed {len(accuracies)}...")

    byte_sizes = np.array(byte_sizes)
    accuracies = np.array(accuracies)
    labels = np.array(labels)

    df = pd.DataFrame(
        {"bytes": byte_sizes, "acc": accuracies, "label": labels}
    )

    return df


def _make_batches(client_tensors, labels, many_func, batch_size):
    xss = []
    lss = []
    bss = []

    for client_tensor, label in zip(client_tensors, labels):
        xs, bs = many_func(client_tensor)
        ls = [label] * len(xs)
        xss.extend(xs)
        lss.extend(ls)
        bss.extend(bs)
        while len(xss) >= batch_size:
            xs_ = np.array(xss[:batch_size])
            ls_ = np.array(lss[:batch_size])
            bs_ = np.array(bss[:batch_size])
            yield xs_, ls_, bs_
            xss = xss[batch_size:]
            lss = lss[batch_size:]
            bss = bss[batch_size:]

    if len(xss) != 0:
        yield np.array(xss), np.array(lss), np.array(bss)


def _generate_jpeg_tensors(
    client_tensor: np.ndarray,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
) -> Tuple[List[np.ndarray], List[int]]:
    """Returns reconstructed tensors from various compressed sizes"""
    tensor_layout = TensorLayout.from_tensor(client_tensor, "hwc")
    tiled_layout = JpegPostencoder(tensor_layout).tiled_layout
    d = {}
    xs = []
    bs = []

    for quality in range(1, 101):
        postencoder = JpegPostencoder(tensor_layout, quality=quality)
        x = client_tensor
        x = quant(x)
        x = postencoder.run(x)
        b = len(x)
        d[b] = x, quality

    for b, (x, quality) in d.items():
        predecoder = JpegPredecoder(tiled_layout, tensor_layout)
        x = predecoder.run(x)
        x = dequant(x)
        xs.append(x)
        bs.append(b)

    return xs, bs


def _bin_for_plot(
    df: pd.DataFrame, key: str, bins: np.ndarray
) -> pd.DataFrame:
    mids = 0.5 * (bins[:-1] + bins[1:])
    df = df.sort_values(key)
    df = df[(bins[0] < df[key]) & (df[key] < bins[-1])]
    idxs = np.digitize(df[key], bins) - 1
    df[key] = mids[idxs]
    return df


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
