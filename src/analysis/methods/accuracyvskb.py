from collections import defaultdict
from os import path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from src.analysis import plot
from src.analysis.utils import predict_dataset
from src.lib.layouts import TensorLayout
from src.lib.postencode import (
    Jpeg2000Postencoder,
    Jpeg2000RgbPostencoder,
    JpegPostencoder,
    JpegRgbPostencoder,
    PngPostencoder,
    PngRgbPostencoder,
    Postencoder,
)
from src.lib.predecode import (
    Jpeg2000Predecoder,
    Jpeg2000RgbPredecoder,
    JpegPredecoder,
    JpegRgbPredecoder,
    PngPredecoder,
    PngRgbPredecoder,
    Predecoder,
)

BYTES_PER_KB = 1000

MakePostencoder = Callable[..., Postencoder]
MakePredecoder = Callable[[Postencoder], Predecoder]
MakePostencoderDecorator = Callable[[MakePostencoder], MakePostencoder]
MakePredecoderDecorator = Callable[[MakePredecoder], MakePredecoder]


def analyze_accuracyvskb_layer(
    model_name: str,
    model: keras.Model,
    model_client: keras.Model,
    model_server: keras.Model,
    dataset: tf.data.Dataset,
    title: str,
    basename: str,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
    postencoder: str,
    batch_size: int,
    subdir: str = "",
    make_postencoder_decorator: Optional[MakePostencoderDecorator] = None,
    make_predecoder_decorator: Optional[MakePredecoderDecorator] = None,
):
    if len(model_client.output_shape) != 4:
        return

    basedir = "img/accuracyvskb"
    shareddir = path.join(basedir, subdir)
    filename_server = path.join(shareddir, f"{model_name}-server.csv")
    filename_shared = path.join(shareddir, f"{basename}-shared.csv")

    bins = np.logspace(0, np.log10(30), num=30) * 1024

    try:
        data_server = pd.read_csv(filename_server)
    except FileNotFoundError:
        args = (model, dataset)
        args = (*args, postencoder, batch_size, bins)
        data_server = _evaluate_accuracies_server_kb(*args)
        data_server.to_csv(filename_server, index=False)
        print("Analyzed server accuracy vs KB")

    try:
        data_shared = pd.read_csv(filename_shared)
    except FileNotFoundError:
        args = (model_client, model_server, dataset, quant, dequant)
        args = (*args, make_postencoder_decorator, make_predecoder_decorator)
        args = (*args, postencoder, batch_size, bins)
        data_shared = _evaluate_accuracies_shared_kb(*args)
        data_shared.to_csv(filename_shared, index=False)
        print("Analyzed shared accuracy vs KB")

    _dataframe_normalize(data_server)
    _dataframe_normalize(data_shared)

    data_server = _bin_for_plot(data_server, "kbs", bins / 1024)
    data_shared = _bin_for_plot(data_shared, "kbs", bins / 1024)

    fig = _plot_accuracy_vs_kb(title, data_server, data_shared)
    plot.save(fig, path.join(shareddir, f"{basename}.png"))
    print("Analyzed accuracy vs KB")


def _evaluate_accuracies_server_kb(
    model: keras.Model,
    dataset: tf.data.Dataset,
    postencoder: str,
    batch_size: int,
    bins: np.ndarray,
) -> pd.DataFrame:
    quant = lambda x: x.astype(np.uint8)
    dequant = lambda x: x.astype(dtype)

    shape = model.input_shape[1:]
    dtype = model.layers[0].dtype
    tensor_layout = TensorLayout.from_shape(shape, "hwc", dtype)

    tensors = tfds.as_numpy(dataset.map(_first))
    labels = np.array(list(tfds.as_numpy(dataset.map(_second))))

    mids = 0.5 * (bins[:-1] + bins[1:])
    make_postencoder, make_predecoder, kwargs_list = server_postencoder_maker(
        postencoder, tensor_layout, mids
    )
    gen_func = lambda x: _generate_tensors(
        x, quant, dequant, make_postencoder, make_predecoder, kwargs_list
    )
    many_func = lambda x: _bin_uniquely(*gen_func(x), bins, mids)

    batches = _make_batches(tensors, labels, many_func, batch_size)
    return _predict_batches(model, batches)


def _evaluate_accuracies_shared_kb(
    model_client: keras.Model,
    model_server: keras.Model,
    dataset: tf.data.Dataset,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
    make_postencoder_decorator: Optional[MakePostencoderDecorator],
    make_predecoder_decorator: Optional[MakePredecoderDecorator],
    postencoder: str,
    batch_size: int,
    bins: np.ndarray,
) -> pd.DataFrame:
    shape = model_client.output_shape[1:]
    dtype = model_client.dtype
    tensor_layout = TensorLayout.from_shape(shape, "hwc", dtype)

    tensors = predict_dataset(model_client, dataset.batch(batch_size))
    labels = np.array(list(tfds.as_numpy(dataset.map(_second))))

    mids = 0.5 * (bins[:-1] + bins[1:])
    make_postencoder, make_predecoder, kwargs_list = shared_postencoder_maker(
        postencoder, tensor_layout, mids
    )
    if make_postencoder_decorator:
        make_postencoder = make_postencoder_decorator(make_postencoder)
    if make_predecoder_decorator:
        make_predecoder = make_predecoder_decorator(make_predecoder)
    gen_func = lambda x: _generate_tensors(
        x, quant, dequant, make_postencoder, make_predecoder, kwargs_list
    )
    many_func = lambda x: _bin_uniquely(*gen_func(x), bins, mids)

    batches = _make_batches(tensors, labels, many_func, batch_size)
    return _predict_batches(model_server, batches)


def _make_batches(
    tensors: Iterator[np.ndarray],
    labels: Iterator[Any],
    many_func: Callable[[np.ndarray], List[np.ndarray]],
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Flatmap many_func into batches.

    Essentially the same as:

    ```
    (
        tensors
        .zip(labels)
        .flatMap(lambda x, l: zip(*(xs, bs := many_func(x)), [l] * len(xs)))
        .batch(batch_size)
    )
    ```
    """
    xss = []
    lss = []
    bss = []

    for tensor, label in zip(tensors, labels):
        xs, bs = many_func(tensor)
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


def _predict_batches(
    model: keras.Model,
    batches: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    byte_sizes = []
    accuracies = []
    labels = []

    for xs, ls, bs in batches:
        xs = model.predict_on_batch(xs)
        accs = _categorical_top1_accuracy(ls, xs)
        accuracies.extend(accs)
        labels.extend(ls)
        byte_sizes.extend(bs)
        print(f"processed {len(accuracies)}...")

    byte_sizes = np.array(byte_sizes)
    accuracies = np.array(accuracies)
    labels = np.array(labels)

    return pd.DataFrame(
        {"bytes": byte_sizes, "acc": accuracies, "label": labels}
    )


def _generate_tensors(
    client_tensor: np.ndarray,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
    make_postencoder: MakePostencoder,
    make_predecoder: MakePredecoder,
    kwargs_list: Iterator[Dict[str, Any]],
) -> Tuple[List[np.ndarray], List[int]]:
    """Returns reconstructed tensors from various compressed sizes"""
    xs = []
    bs = []

    for kwargs in kwargs_list:
        postencoder = make_postencoder(**kwargs)
        x = client_tensor
        x = quant(x)
        x = postencoder.run(x)
        b = len(x)
        if b in bs:
            continue
        predecoder = make_predecoder(postencoder)
        x = predecoder.run(x)
        x = dequant(x)
        xs.append(x)
        bs.append(b)

    return xs, bs


def _bin_uniquely(xs, bs, bins, mids) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only one value per bin"""
    idxs = np.digitize(bs, bins) - 1
    d = defaultdict(list)
    for idx, x, b in zip(idxs, xs, bs):
        if idx >= len(bins) - 1:
            continue
        d[idx].append((x, b))
    xbs = [
        min(xbs, key=lambda xb, i=i: abs(xb[1] - mids[i]))
        for i, xbs in d.items()
    ]
    xs = np.array([x for x, b in xbs])
    bs = np.array([b for x, b in xbs])
    return xs, bs


def _dataframe_normalize(df: pd.DataFrame):
    if "kbs" not in df.columns:
        df["kbs"] = df["bytes"] / 1024
        df.drop("bytes", axis=1, inplace=True)


def _bin_for_plot(df: pd.DataFrame, key: str, bins: np.ndarray):
    df.sort_values(key, inplace=True)

    drop_mask = (df[key] <= bins[0]) | (bins[-1] <= df[key])
    df.drop(df[drop_mask].index, inplace=True)

    idxs = np.digitize(df[key], bins) - 1
    mids = 0.5 * (bins[:-1] + bins[1:])
    df[key] = mids[idxs]

    min_samples = 900
    df.set_index(key, inplace=True)
    counts = df.groupby(key).size()
    df.drop(counts[counts < min_samples].index, inplace=True)
    df.reset_index(inplace=True)

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
    ax.set_xlabel("Compressed size (KB)")
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


def server_postencoder_maker(
    postencoder: str, tensor_layout: TensorLayout, mids: np.ndarray,
) -> Tuple[MakePostencoder, MakePredecoder, List[dict]]:
    return {
        "jpeg": (
            JpegRgbPostencoder,
            lambda _p: JpegRgbPredecoder(tensor_layout),
            [{"quality": quality} for quality in range(1, 101)],
        ),
        "jpeg2000": (
            Jpeg2000RgbPostencoder,
            lambda _p: Jpeg2000RgbPredecoder(tensor_layout),
            [{"out_size": out_size} for out_size in mids],
        ),
        "png": (
            PngRgbPostencoder,
            lambda _p: PngRgbPredecoder(tensor_layout),
            [{}],
        ),
    }[postencoder.lower()]


def shared_postencoder_maker(
    postencoder: str, tensor_layout: TensorLayout, mids: np.ndarray,
) -> Tuple[MakePostencoder, MakePredecoder, List[dict]]:
    return {
        "jpeg": (
            lambda **kwargs: JpegPostencoder(tensor_layout, **kwargs),
            lambda p: JpegPredecoder(p.tiled_layout, p.tensor_layout),
            [{"quality": quality} for quality in range(1, 101)],
        ),
        "jpeg2000": (
            lambda **kwargs: Jpeg2000Postencoder(tensor_layout, **kwargs),
            lambda p: Jpeg2000Predecoder(p.tiled_layout, p.tensor_layout),
            [{"out_size": out_size} for out_size in mids],
        ),
        "png": (
            lambda **kwargs: PngPostencoder(tensor_layout, **kwargs),
            lambda p: PngPredecoder(p.tiled_layout, p.tensor_layout),
            [{}],
        ),
    }[postencoder.lower()]
