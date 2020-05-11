import itertools
from functools import partial
from os import path
from typing import Callable, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import LogNorm

from src.analysis import plot
from src.analysis.experimentrunner import ExperimentRunner
from src.analysis.methods.accuracyvskb import (
    _bin_for_plot,
    _dataframe_normalize,
    _evaluate_accuracies_server_kb,
    _evaluate_accuracies_shared_kb,
    _plot_accuracy_vs_kb,
)
from src.analysis.methods.stats import analyze_stats_layer
from src.analysis.quant import *
from src.analysis.utils import (
    categorical_top1_accuracy,
    tf_disable_eager_execution,
)
from src.lib.postencode import (
    CallablePostencoder,
    JpegPostencoder,
    PngPostencoder,
)
from src.lib.predecode import (
    CallablePredecoder,
    JpegPredecoder,
    PngPredecoder,
)
from src.lib.tile import determine_tile_layout, tile

tf_disable_eager_execution()

BATCH_SIZE = 64
# BATCH_SIZE = 512
# DATASET_SIZE = 64
# DATASET_SIZE = 1024
# DATASET_SIZE = 2048
DATASET_SIZE = 4096
# DATASET_SIZE = 8192


def load_array(filename, f, *args, dtype=None) -> np.ndarray:
    try:
        z = np.load(filename)
    except FileNotFoundError:
        z = np.empty([x.size for x in args], dtype=dtype)
        pairs = [list(enumerate(xs)) for xs in args]
        for xs in itertools.product(*pairs):
            idx = tuple(x[0] for x in xs)
            arg = tuple(x[1] for x in reversed(xs))
            z[idx] = f(*arg)
        np.save(filename, z)
    return z


def load_2d_array(filename, f, x, y, dtype=None) -> np.ndarray:
    try:
        z = np.load(filename)
    except FileNotFoundError:
        z = np.empty((y.size, x.size), dtype=dtype)
        for i, xi in enumerate(x):
            # print(f"i={i}, xi={xi}")
            for j, yi in enumerate(y):
                z[j, i] = f(xi, yi)
        np.save(filename, z)
    return z


def plot_accuracy_quant(
    runner: ExperimentRunner, compute_func: Callable[..., float], suffix: str,
):
    plot_levels = 20
    cmin, cmax = 0.8, 1.0
    cmap = plot.colormap_upper(
        cmin=cmin, cmax=cmax, levels=plot_levels, gamma=16.0
    )

    xs = np.linspace(1, 5, num=50)
    ys = np.arange(2, 20)

    path = f"img/experiment/{runner.basename}-{suffix}.npy"
    accuracies = load_2d_array(path, partial(compute_func, runner), xs, ys)
    accuracies = np.clip(accuracies, cmin, cmax)

    x, y = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(tight_layout=True)
    cs = ax.contourf(
        x, y, accuracies, cmap=cmap, levels=plot_levels, vmin=cmin, vmax=cmax
    )
    cs.set_clim(cmin, cmax)
    cbar = fig.colorbar(cs, ticks=np.linspace(cmin, cmax, 6))
    cbar.ax.set_title("Accuracy", fontsize="small")
    ax.set_xlabel(r"Clip width ($\sigma$)")
    ax.set_ylabel("Quantization levels")
    ax.set_title(runner.title)
    save_kwargs = dict(dpi=300, bbox_inches="tight")
    path = f"img/experiment/{runner.basename}-{suffix}.png"
    fig.savefig(path, **save_kwargs)


def plot_mse_quant(
    runner: ExperimentRunner, compute_func: Callable[..., float], suffix: str,
):
    xs = np.linspace(1, 5, num=50)
    ys = np.arange(2, 20)

    path = f"img/experiment/{runner.basename}-{suffix}.npy"
    mses = load_2d_array(path, partial(compute_func, runner), xs, ys)

    x, y = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(tight_layout=True)
    # norm = LogNorm(vmin=mses.min(), vmax=mses.max())
    norm = LogNorm()
    plot_levels = np.logspace(-3, 1, 4 * 4 + 1)
    cs = ax.contourf(x, y, mses, norm=norm, levels=plot_levels)
    cbar = fig.colorbar(cs)
    # cbar = fig.colorbar(cs, ticks=np.logspace(-2, 1, n=5),
    #     format=ticker.LogFormatter(10, labelOnlyBase=True))
    cbar.ax.set_title("MSE", fontsize="small")
    ax.set_xlabel(r"Clip width ($\sigma$)")
    ax.set_ylabel("Quantization levels")
    ax.set_title(runner.title)
    save_kwargs = dict(dpi=300, bbox_inches="tight")
    path = f"img/experiment/{runner.basename}-{suffix}.png"
    fig.savefig(path, **save_kwargs)


def plot_distorted(
    runner: ExperimentRunner,
    compute_func: Callable[..., float],
    suffix: str,
    xs: np.ndarray,
    xlabel: str,
):
    path = f"img/experiment/{runner.basename}-{suffix}.npy"
    accuracies = load_array(path, compute_func, xs)
    accuracies = np.clip(accuracies, 0.0, 1.0)
    x, y = xs, accuracies
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(runner.title)
    save_kwargs = dict(dpi=300, bbox_inches="tight")
    path = f"img/experiment/{runner.basename}-{suffix}.png"
    fig.savefig(path, **save_kwargs)


def plot_distorted_agg(runner, data, suffix, xlabel, ylabel):
    fig, ax = plt.subplots(tight_layout=True)
    for x, y, label in data:
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(runner.title)
    ax.legend(labels=[label for _, _, label in data], loc="upper right")
    save_kwargs = dict(dpi=300, bbox_inches="tight")
    path = f"img/experiment/{runner.basename}-{suffix}.png"
    fig.savefig(path, **save_kwargs)


def plot_distorted_2d(
    runner: ExperimentRunner,
    compute_func: Callable[..., float],
    suffix: str,
    xs: np.ndarray,
    ys: np.ndarray,
    xlabel: str,
    ylabel: str,
):
    path = f"img/experiment/{runner.basename}-{suffix}.npy"
    accuracies = load_array(path, compute_func, xs, ys)
    accuracies = np.clip(accuracies, 0.0, 1.0)
    x, y = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(tight_layout=True)
    cs = ax.contourf(x, y, accuracies, levels=256)
    cbar = fig.colorbar(cs, ticks=np.linspace(0, 1, 6))
    cbar.ax.set_title("Accuracy", fontsize="small")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(runner.title)
    save_kwargs = dict(dpi=300, bbox_inches="tight")
    path = f"img/experiment/{runner.basename}-{suffix}.png"
    fig.savefig(path, **save_kwargs)


def compute_stats(runner: ExperimentRunner):
    basename = f"img/experiment/{runner.basename}"

    def plot_quick(plot_func, tensor, suffix, **kwargs):
        fig = plot_func(tensor, runner.title, **kwargs)
        plot.save(fig, f"{basename}-{suffix}.png", bbox_inches="tight")

    d = runner.d
    client_tensors = runner.data.client_tensors

    x = client_tensors[0]
    m = d["mean"]
    w = d["std"] * 3
    x = uni_quant(x, (m - w, m + w), 4)
    plot_quick(plot.featuremap, x, "uniquant_featuremap")

    x = client_tensors[0]
    x = indep_quant(x, d, 3, 4)
    # x = indep_dequant(x, d, 3, 4)
    plot_quick(plot.featuremap, x, "indepquant_featuremap")

    x = client_tensors[0]
    tensor_layout = runner.tensor_layout
    tiled_layout = determine_tile_layout(tensor_layout)
    xt = tile(x, tensor_layout, tiled_layout)
    np.save("img/experiment/sample.npy", x)
    uniquant2 = uni_quant(xt, (m - w, m + w), 2)
    np.save("img/experiment/sample_uniquant2.npy", uniquant2)
    uniquant4 = uni_quant(xt, (m - w, m + w), 4)
    np.save("img/experiment/sample_uniquant4.npy", uniquant4)
    uniquant8 = uni_quant(xt, (m - w, m + w), 8)
    np.save("img/experiment/sample_uniquant8.npy", uniquant8)
    uniquant256 = uni_quant(xt, (m - w, m + w), 256)
    np.save("img/experiment/sample_uniquant256.npy", uniquant256)

    n = client_tensors.size
    d["pct_clipped"] = np.count_nonzero((client_tensors - m) > w) / n
    x = client_tensors
    x = uni_quant(x, (m - w, m + w), 256)
    x = uni_dequant(x, (m - w, m + w), 256)
    pred_tensors = runner.model_server.predict(x)
    d["accuracy_256"] = np.mean(
        categorical_top1_accuracy(labels, pred_tensors)
    )
    runner.d.update(d)


def compute_accuracy(
    runner: ExperimentRunner, quant, dequant, *args, **kwargs
) -> float:
    accs = []
    data = runner.data.client_tensor_batches(copy=False)
    for client_tensors, labels in data:
        quant_tensors = quant(client_tensors, *args, **kwargs)
        recv_tensors = dequant(quant_tensors, *args, **kwargs)
        pred_tensors = runner.model_server.predict(recv_tensors)
        accs.extend(categorical_top1_accuracy(labels, pred_tensors))
    max_accuracy = runner.d["accuracy"]
    return np.mean(accs) / max_accuracy


def compute_mse(
    runner: ExperimentRunner, quant, dequant, *args, **kwargs
) -> float:
    mses = []
    data = runner.data.client_tensor_batches(copy=False)
    for client_tensors, _labels in data:
        quant_tensors = quant(client_tensors, *args, **kwargs)
        recv_tensors = dequant(quant_tensors, *args, **kwargs)
        mses.extend(((recv_tensors - client_tensors) ** 2).mean(axis=0))
    return np.mean(mses)


def compute_accuracy_distort(runner: ExperimentRunner, distort=None) -> float:
    accs = []
    data = runner.data.client_tensor_batches()
    for client_tensors, labels in data:
        if distort:
            for i, _ in enumerate(client_tensors):
                client_tensors[i] = distort(client_tensors[i])
        pred_tensors = runner.model_server.predict(client_tensors)
        accs.extend(categorical_top1_accuracy(labels, pred_tensors))
    max_accuracy = runner.d["accuracy"]
    return np.mean(accs) / max_accuracy


def compute_distort(runner: ExperimentRunner, distort, *distort_args):
    def distort_(x: np.ndarray) -> np.ndarray:
        return distort(x, *distort_args)

    return compute_accuracy_distort(runner, distort_)


def importance_blacken(
    runner: ExperimentRunner,
    data: Iterator[Tuple[np.ndarray, np.ndarray]],
    blacken_channel: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Find important channels by calculating channel-blackened accuracy."""
    channels = runner.tensor_layout.c
    accuracies = [[] for _ in range(channels)]
    count = 0

    for client_tensors, labels in data:
        # print(f"tensor {count}")
        for client_tensor, label in zip(client_tensors, labels):
            shape = (channels, *client_tensor.shape)
            xs = np.empty(shape, dtype=client_tensor.dtype)
            for c in range(channels):
                x = np.array(client_tensor)
                xs[c] = client_tensor
                xs[c, ..., c] = blacken_channel(x, c)[..., c]
            preds = runner.model_server.predict(xs, batch_size=BATCH_SIZE)
            accs = categorical_top1_accuracy(label, preds)
            for c in range(channels):
                accuracies[c].append(accs[c])
            count += 1

    accuracies_avg = np.empty(channels)
    for c in range(runner.tensor_layout.c):
        accuracies_avg[c] = np.array(accuracies[c]).mean()

    return accuracies_avg


def blacken_single_channel(
    runner, basepath, name, func, importance_func, clim,
):
    npy_path = f"{basepath}_{name}.npy"
    data = runner.data.client_tensor_batches(copy=False)
    try:
        importance = np.load(npy_path)
    except FileNotFoundError:
        importance = importance_func(runner, data, func)
        np.save(npy_path, importance)
    importance = importance / runner.d["accuracy"]
    importance = np.clip(importance, 0.0, 1.0)
    featuremap = np.broadcast_to(importance, runner.tensor_layout.shape)
    cmap = plot.colormap_upper(levels=256, gamma=2.0)
    fig = plot.featuremap(featuremap, runner.title, clim=clim, cmap=cmap)
    plot.save(fig, f"{basepath}_{name}.png", bbox_inches="tight")


def blacken_data(
    runner: ExperimentRunner,
    data: Iterator[Tuple[np.ndarray, np.ndarray]],
    channel_sequence: List[int],
    func: Callable[[np.ndarray, int], np.ndarray],
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for client_tensors, labels in data:
        for i in range(len(client_tensors)):
            for c in channel_sequence:
                x = np.array(client_tensors[i])
                black_channel = func(x, c)[..., c]
                client_tensors[i, ..., c] = black_channel
        yield client_tensors, labels


def whiten_data(
    runner: ExperimentRunner,
    data: Iterator[Tuple[np.ndarray, np.ndarray]],
    channel_sequence: List[int],
    func: Callable[[np.ndarray, int], np.ndarray],
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    seq = list(set(range(runner.tensor_layout.c)) - set(channel_sequence))
    return blacken_data(runner, data, seq, func)


def best_blacken_channel_sequence(
    runner,
    basepath,
    name,
    func,
    data_func,
    importance_func,
    nanoptimize_func,
    clim,
) -> List[int]:
    channel_sequence = []
    exclude = np.zeros(runner.tensor_layout.c, dtype=np.bool)

    for i in range(runner.tensor_layout.c):
        basefilename = f"{basepath}_{name}_{i}"
        data = runner.data.client_tensor_batches()
        data = data_func(runner, data, channel_sequence)
        importance = importance_func(runner, data, func)
        importance_excluded = np.array(importance)
        importance_excluded[exclude] = np.NaN
        channel = nanoptimize_func(importance_excluded)
        channel_sequence.append(channel)
        exclude[channel] = True
        data = runner.data_test.client_tensor_batches()
        data = data_func(runner, data, channel_sequence)
        importance = importance_func(runner, data, func)
        np.save(f"{basefilename}.npy", importance)
        importance = importance / runner.d["accuracy"]
        importance = np.clip(importance, 0.0, 1.0)
        featuremap = np.broadcast_to(importance, runner.tensor_layout.shape)
        title = f"{runner.title} ({channel})"
        cmap = plot.colormap_upper(levels=256, gamma=2.0)
        fig = plot.featuremap(featuremap, title, clim=clim, cmap=cmap)
        plot.save(fig, f"{basefilename}.png", bbox_inches="tight")
        print(f"channel sequence: {channel_sequence}")

    return channel_sequence


def importance_whiten(
    runner: ExperimentRunner,
    data: Iterator[Tuple[np.ndarray, np.ndarray]],
    blacken_channel: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Find important channels by calculating channel-whitened accuracy."""
    channels = runner.tensor_layout.c
    accuracies = [[] for _ in range(channels)]
    count = 0

    for client_tensors, labels in data:
        # print(f"tensor {count}")
        for client_tensor, label in zip(client_tensors, labels):
            shape = (channels, *client_tensor.shape)
            black = np.empty_like(client_tensor)
            for c in range(channels):
                x = np.array(client_tensor)
                black[..., c] = blacken_channel(x, c)[..., c]
            xs = np.empty(shape, dtype=client_tensor.dtype)
            for c in range(channels):
                xs[c] = black
                xs[c, ..., c] = client_tensor[..., c]
            preds = runner.model_server.predict(xs, batch_size=BATCH_SIZE)
            accs = categorical_top1_accuracy(label, preds)
            for c in range(channels):
                accuracies[c].append(accs[c])
            count += 1

    accuracies_avg = np.empty(channels)
    for c in range(runner.tensor_layout.c):
        acc = np.array(accuracies[c]).mean()
        accuracies_avg[c] = acc / runner.d["accuracy"]
    accuracies_avg = np.clip(accuracies_avg, 0.0, 1.0)

    return accuracies_avg


def analyze_accuracyvskb_layer(
    runner: ExperimentRunner,
    quant: Callable[[np.ndarray], np.ndarray],
    dequant: Callable[[np.ndarray], np.ndarray],
    postencoder: str,
    subdir: str = "",
):
    if len(runner.model_client.output_shape) != 4:
        return

    title = runner.title
    basename = runner.basename
    basedir = "img/accuracyvskb"
    shareddir = path.join(basedir, subdir)
    filename_server = path.join(basedir, f"{runner.model_name}-server.csv")
    filename_shared = path.join(shareddir, f"{runner.basename}-shared.csv")

    try:
        data_server = pd.read_csv(filename_server)
    except FileNotFoundError:
        data_server = _evaluate_accuracies_server_kb(
            runner.model, runner.data.batch_size
        )
        data_server.to_csv(filename_server, index=False)
        print("Analyzed server accuracy vs KB")

    try:
        data_shared = pd.read_csv(filename_shared)
    except FileNotFoundError:
        data_shared = _evaluate_accuracies_shared_kb(
            runner.model_client,
            runner.model_server,
            quant,
            dequant,
            postencoder,
            runner.data.batch_size,
        )
        data_shared.to_csv(filename_shared, index=False)
        print("Analyzed shared accuracy vs KB")

    _dataframe_normalize(data_server)
    _dataframe_normalize(data_shared)

    bins = np.logspace(0, np.log10(30), num=30)
    data_shared = _bin_for_plot(data_shared, "kbs", bins)

    fig = _plot_accuracy_vs_kb(title, data_server, data_shared)
    plot.save(fig, path.join(shareddir, f"{basename}.png"))
    print("Analyzed accuracy vs KB")


def quick_accuracy_test(runner: ExperimentRunner):
    # TODO scale? (easier quantization for JPEG...?)
    print("Quick codec test")
    quality = 40
    bits = 8
    m = runner.d["mean"]
    w = runner.d["std"] * 3
    clip_range = (m - w, m + w)
    levels = 2 ** bits
    scale = 1
    # scale = 2**(8 - bits)
    # qu = lambda x: indep_quant(x, runner.d, 3, levels)
    # dq = lambda x: indep_dequant(x, runner.d, 3, levels)
    qu = lambda x: uni_quant(x, clip_range, levels)
    dq = lambda x: uni_dequant(x, clip_range, levels)
    quant = lambda x: (qu(x) * scale).astype(np.uint8)
    dequant = lambda x: (dq(x) / scale).astype(np.float32)
    accuracy_func = categorical_top1_accuracy
    tensor_layout = runner.tensor_layout
    postencoder_img = JpegPostencoder(tensor_layout, quality=quality)
    # postencoder_img = PngPostencoder(tensor_layout, bits=bits)
    tiled_layout = postencoder_img.tiled_layout
    postencoder = CallablePostencoder(lambda x: postencoder_img.run(quant(x)))
    predecoder_img = JpegPredecoder(tiled_layout, tensor_layout)
    # predecoder_img = PngPredecoder(tiled_layout, tensor_layout)
    predecoder = CallablePredecoder(lambda x: dequant(predecoder_img.run(x)))
    accuracies = []
    kbs = []
    data = runner.data.client_tensor_batches(images=True, copy=False)
    for client_tensors, _frames, labels in data:
        # client_tensors = runner.model_client.predict(frames)
        decoded_tensors = []
        for client_tensor in client_tensors:
            encoded_bytes = postencoder.run(client_tensor)
            decoded_tensor = predecoder.run(encoded_bytes)
            decoded_tensors.append(decoded_tensor)
            kbs.append(len(encoded_bytes) / 1024)
        decoded_tensors = np.array(decoded_tensors)
        predictions = runner.model_server.predict(decoded_tensors)
        # labels = labels.numpy()
        accuracies.extend(accuracy_func(labels, predictions))
    accuracy = np.mean(accuracies)
    kb = np.mean(kbs)  # TODO this doesn't really have that much meaning
    print(f"accuracy: {accuracy:.3g}")
    print(f"acc_drop: {1 - accuracy / runner.d['accuracy']:.3g}")
    print(f"kb_mean: {kb:.3g}")


def main():
    # TODO gaussian window? hmmm
    def distort_zeroing(x: np.ndarray, win_mid, win_width) -> np.ndarray:
        x = x.copy()
        mask = (x > win_mid - win_width) & (x < win_mid + win_width)
        x[mask] = runner.d["mean"]
        # x[mask] = runner.d["mean"] + 3 * runner.d["std"]  # lol
        # x[mask] = 0
        # x[mask] = win_mid
        # x[mask] = runner.d["tensors_mean"][mask]
        return x

    def random_mask(pct: float, shape: Tuple[int]) -> np.ndarray:
        """Returns mask of given shape with pct of values set to True."""
        n = pct * np.prod(shape)
        mask = np.random.rand(*shape)
        v = np.partition(mask.flatten(), int(n))[int(n)]
        mask = mask < v  # TODO this should maybe be int(n) - 1 ... or <= ...
        if np.random.rand() >= n - int(n):
            return mask
        while True:
            idx = tuple(np.random.randint(0, i) for i in shape)
            if mask[idx]:
                continue
            mask[idx] = True
            break
        return mask

    def black_neuron_pr(x: np.ndarray, black: np.ndarray, p) -> np.ndarray:
        mask = random_mask(p, x.shape)
        x = x.copy()
        x[mask] = black[mask]
        return x

    def black_channel_pr(x: np.ndarray, black: np.ndarray, p) -> np.ndarray:
        mask = random_mask(p, x.shape[:-1])
        x = x.copy()
        x[mask] = black[mask]
        return x

    def distort_black_neuron_zero(x: np.ndarray, p) -> np.ndarray:
        black = np.zeros_like(x)
        return black_neuron_pr(x, black, p)

    def distort_black_channel_zero(x: np.ndarray, p) -> np.ndarray:
        black = np.zeros_like(x)
        return black_channel_pr(x, black, p)

    def distort_black_neuron_tensorsmean(x: np.ndarray, p) -> np.ndarray:
        black = runner.d["tensors_mean"]
        return black_neuron_pr(x, black, p)

    def distort_black_channel_tensorsmean(x: np.ndarray, p) -> np.ndarray:
        black = runner.d["tensors_mean"]
        return black_channel_pr(x, black, p)

    def distort_black_neuron_channelmean(x: np.ndarray, p) -> np.ndarray:
        black = np.broadcast_to(np.mean(x, axis=-1)[..., np.newaxis], x.shape)
        return black_neuron_pr(x, black, p)

    def distort_black_channel_channelmean(x: np.ndarray, p) -> np.ndarray:
        black = np.broadcast_to(np.mean(x, axis=-1)[..., np.newaxis], x.shape)
        return black_channel_pr(x, black, p)

    def distort_black_neuron_tensoroffsetmean(x: np.ndarray, p) -> np.ndarray:
        mu = np.broadcast_to(np.mean(x, axis=-1)[..., np.newaxis], x.shape)
        tm = runner.d["tensors_mean"]
        black = mu + tm - np.mean(tm, axis=-1)[..., np.newaxis]
        return black_neuron_pr(x, black, p)

    def distort_black_channel_tensoroffsetmean(x: np.ndarray, p) -> np.ndarray:
        mu = np.broadcast_to(np.mean(x, axis=-1)[..., np.newaxis], x.shape)
        tm = runner.d["tensors_mean"]
        black = mu + tm - np.mean(tm, axis=-1)[..., np.newaxis]
        return black_channel_pr(x, black, p)

    def acc_uniquant(runner: ExperimentRunner, width_sigma, levels):
        m, ws = runner.d["mean"], width_sigma * runner.d["std"]
        clip_range = (m - ws, m + ws)
        args = clip_range, levels
        return compute_accuracy(runner, uni_quant, uni_dequant, *args)

    def acc_indepquant(runner: ExperimentRunner, width_sigma, levels):
        args = runner.d, width_sigma, levels
        return compute_accuracy(runner, indep_quant, indep_dequant, *args)

    def acc_qcut(runner: ExperimentRunner, width_sigma, levels):
        m, ws = runner.d["mean"], width_sigma * runner.d["std"]
        clip_range = (m - ws, m + ws)
        bins = qcut_bins(runner.d["tensors"].clip(*clip_range), levels)
        return compute_accuracy(runner, bin_quant, bin_dequant, bins)

    def mse_uniquant(runner: ExperimentRunner, width_sigma, levels):
        m, ws = runner.d["mean"], width_sigma * runner.d["std"]
        clip_range = (m - ws, m + ws)
        return compute_mse(runner, uni_quant, uni_dequant, clip_range, levels)

    def mse_indepquant(runner: ExperimentRunner, width_sigma, levels):
        args = runner.d, width_sigma, levels
        return compute_mse(runner, indep_quant, indep_dequant, *args)

    def mse_qcut(runner: ExperimentRunner, width_sigma, levels):
        m, ws = runner.d["mean"], width_sigma * runner.d["std"]
        clip_range = (m - ws, m + ws)
        bins = qcut_bins(runner.d["tensors"].clip(*clip_range), levels)
        return compute_mse(runner, bin_quant, bin_dequant, bins)

    def acc_zero(runner: ExperimentRunner, width_sigma):
        win_mid = runner.d["mean"]
        win_width = width_sigma * runner.d["std"]
        return compute_distort(runner, distort_zeroing, win_mid, win_width)

    def acc_zero_2d(runner: ExperimentRunner, center_sigma, width_sigma):
        win_mid = runner.d["mean"] + center_sigma * runner.d["std"]
        win_width = width_sigma * runner.d["std"]
        return compute_distort(runner, distort_zeroing, win_mid, win_width)

    # def acc_black(runner: ExperimentRunner, probability):
    #     return compute_distort(runner, distort_black, probability)

    def blacken_channel_tensorsmean(x: np.ndarray, c: int) -> np.ndarray:
        m = runner.d["tensors_mean"]
        x[..., c] = m[..., c]
        return x

    def blacken_channel_channelmean(x: np.ndarray, c: int) -> np.ndarray:
        x[..., c] = x[..., c].mean()
        return x

    def blacken_channel_uniquant(
        x: np.ndarray, c: int, levels: int
    ) -> np.ndarray:
        m = runner.d["mean"]
        s = runner.d["std"]
        clip_range = (m - 3 * s, m + 3 * s)
        x = uni_quant(x, clip_range, levels)
        x = uni_dequant(x, clip_range, levels)
        return x

    def blacken_channel_uniquant4(x: np.ndarray, c: int) -> np.ndarray:
        return blacken_channel_uniquant(x, c, levels=4)

    def blacken_channel_uniquant7(x: np.ndarray, c: int) -> np.ndarray:
        return blacken_channel_uniquant(x, c, levels=7)

    def blacken_channel_uniquant8(x: np.ndarray, c: int) -> np.ndarray:
        return blacken_channel_uniquant(x, c, levels=8)

    runner = ExperimentRunner(
        model_name="resnet34",
        layer_name="add_3",
        # layer_name="stage2_unit1_bn1",
        # layer_name="stage3_unit1_bn1",
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )

    # TODO shouldn't compute stats be part of experimentrunner?
    print("Computing stats...")
    analyze_stats_layer(runner)
    compute_stats(runner)
    runner.summarize()

    print("Uniform quantization...")
    plot_accuracy_quant(runner, acc_uniquant, "uniquant")
    plot_mse_quant(runner, mse_uniquant, "uniquant_mse")

    # TODO optimize...
    # print("Qcut quantization...")
    # plot_accuracy_quant(runner, acc_qcut, "qcutquant")
    # plot_mse_quant(runner, mse_qcut, "qcutquant_mse")

    print("Independent quantization...")
    plot_accuracy_quant(runner, acc_indepquant, "indepquant")
    plot_mse_quant(runner, mse_indepquant, "indepquant_mse")

    print("Importance...")
    shape = runner.tensor_layout.shape
    trials = [
        {"name": "tensorsmean", "func": blacken_channel_tensorsmean},
        {"name": "channelmean", "func": blacken_channel_channelmean},
        {"name": "uniquant4", "func": blacken_channel_uniquant4},
        {"name": "uniquant7", "func": blacken_channel_uniquant7},
        {"name": "uniquant8", "func": blacken_channel_uniquant8},
    ]

    basepath = f"img/experiment/{runner.basename}-importance_black"
    for trial in trials:
        name = trial["name"]
        func = trial["func"]
        print(f"\n\nblack {name}\n")
        blacken_single_channel(
            runner, basepath, name, func, importance_blacken, clim=(0.9, 1.0)
        )

    basepath = f"img/experiment/{runner.basename}-importance_white"
    for trial in trials:
        name = trial["name"]
        func = trial["func"]
        print(f"\n\nwhite {name}\n")
        blacken_single_channel(
            runner, basepath, name, func, importance_whiten, clim=(0.0, 1.0)
        )

    basepath = f"img/experiment/{runner.basename}-importance_black_greedy"
    for trial in trials:
        name = trial["name"]
        func = trial["func"]
        print(f"\n\nblack greedy {name}\n")
        data_func = partial(blacken_data, func=func)
        best_blacken_channel_sequence(
            runner,
            basepath,
            name,
            func,
            data_func,
            importance_blacken,
            np.nanargmax,
            clim=(0.9, 1.0),
        )

    basepath = f"img/experiment/{runner.basename}-importance_white_greedy"
    for trial in trials:
        name = trial["name"]
        func = trial["func"]
        print(f"\n\nwhite greedy {name}\n")
        data_func = partial(whiten_data, func=func)
        best_blacken_channel_sequence(
            runner,
            basepath,
            name,
            func,
            data_func,
            importance_whiten,  # TODO is this... correct? use blacken instead?
            # TODO ah, the problem occurs because this is called on partly
            # blackened tensor data... thus, the original client_tensor data is
            # missing when this is called
            np.nanargmin,
            clim=(0.0, 1.0),
        )
        # TODO shouldn't the sequence be reversed?

    # TODO compare with other curves (tensors_mean)

    # TODO generate sequence in a more robust manner:
    # keep shuffling dataset over larger dataset, but keep separate from
    # test... also shuffle test dataset over larger test dataset

    # TODO save importance arrays to single csv pandas file...

    # TODO blacken_channel affects entire tensor...?
    # Why doesn't it only operate on single given tensor area? hmmmmmm fixable

    # TODO uniquant4 on WHOLE tensor instead of just blackening...?
    # uniquant tensor black=level=2, white=level=7?

    # TODO Find importance sequence? (with lower quantization levels/mean)

    print("Accuracy vs KB...")
    postencoder_name = "jpeg2000"
    subdir = f"{postencoder_name}_uniquant256/{runner.model_name}"
    analyze_accuracyvskb_layer(
        runner, uni_quant, uni_dequant, postencoder_name, subdir
    )

    print("Distort zero...")
    width_sigmas = np.linspace(0, 4, num=20)
    plot_distorted(
        runner,
        partial(acc_zero, runner),
        "distortzero",
        width_sigmas,
        xlabel=r"Clip width ($\sigma$)",
    )

    print("Distort zero 2D...")
    center_sigmas = np.linspace(-4, 4, num=20)
    width_sigmas = np.linspace(0, 4, num=20)
    plot_distorted_2d(
        runner,
        partial(acc_zero_2d, runner),
        "distortzero2d",
        center_sigmas,
        width_sigmas,
        xlabel=r"Window center ($\sigma$)",
        ylabel=r"Window width ($\sigma$)",
    )
    # TODO this should use the per-neuron scaled distribution...

    print("Increasing loss of signal...")
    # black out by channel, or by LSBs, etc; black randomly?
    # uniquant? still haven't determined if that's the best... meh
    probs = np.linspace(0, 1, num=20)
    black_kwargs = dict(runner=runner, xs=probs, xlabel="Probability")
    trials = [
        {
            "func": distort_black_neuron_zero,
            "name": "distort_black_neuron_zero",
        },
        {
            "func": distort_black_neuron_tensorsmean,
            "name": "distort_black_neuron_tensorsmean",
        },
        {
            "func": distort_black_neuron_channelmean,
            "name": "distort_black_neuron_channelmean",
        },
        {
            "func": distort_black_neuron_tensoroffsetmean,
            "name": "distort_black_neuron_tensoroffsetmean",
        },
        {
            "func": distort_black_channel_zero,
            "name": "distort_black_channel_zero",
        },
        {
            "func": distort_black_channel_tensorsmean,
            "name": "distort_black_channel_tensorsmean",
        },
        {
            "func": distort_black_channel_channelmean,
            "name": "distort_black_channel_channelmean",
        },
        {
            "func": distort_black_channel_tensoroffsetmean,
            "name": "distort_black_channel_tensoroffsetmean",
        },
    ]
    for trial in trials:
        plot_distorted(
            compute_func=partial(compute_distort, runner, trial["func"]),
            suffix=trial["name"],
            **black_kwargs,
        )
        # TODO save dataframe with assosciated probs labels instead of npys...

    for prefix in ["distort_black_neuron_", "distort_black_channel_"]:
        data = []
        for trial in trials:
            if not trial["name"].startswith(prefix):
                continue
            suffix = trial["name"][len(prefix) :]
            npy_path = f"img/experiment/{runner.basename}-{prefix}{suffix}.npy"
            accs = np.load(npy_path)
            data.append((probs, accs, suffix))
        suffix = prefix.rstrip("_")
        plot_distorted_agg(runner, data, suffix, "Probability", "Accuracy")

    print("Increasing errors in signal...")
    # plot_distorted(runner, flip_bits)

    # % TODO also look into reducing the number of levels for neurons that don't
    # % need as many levels...!

    # distort with and without quantization? (uniquant 3stddev, 256 levels as
    # default?)

    # TODO black increasing amounts of the tensor
    # Distortion
    # tensor = black_randomly(tensor)
    # then measure accuracy?

    # TODO black MSB/etc

    # sooooo... does blacking to mean make things better or what

    # TODO try distorting original image...?
    # what information does that give, though?

    # what happens if we "black" in the same manner as dropout (randomly)

    # !!!!!! why not normalize each channel via given mean/stddev per channel

    # also, how can we do the N channel similarity thing? maybe try subtracting channels from each other... or maybe manually identify which channels are similar to each other...

    # TODO black input? use mean for "black"? compare with other methods of blacking

    # TODO plot curves for multiple series in plot...

    # TODO
    # Questions/ideas:
    # Are some neurons more important than others?
    #   (If a neuron is not very important towards accuracy, don't give it
    #   that much resolution. Does this hold in a conv net though?)
    #   (Perhaps measure by taking a group of neurons (e.g. with low std), and
    #   giving them less levels. Especially if the "indep_quant" doesn't seem
    #   to give "good" results.)
    # Are some channels more important than others?
    #   (If a channel is not very important towards accuracy, don't give it
    #   that much resolution.)
    # Error redundancy and compression are kind of opposites.
    # Do neurons follow predictable distributions?
    #   (If a neuron/channel is usually within a certain interval, we can
    #   quantize within that interval for better reconstruction.)
    #   (If a neuron/channel shares similar values as its neighbors
    #   [e.g. median filter], then that can be used for prediction.)
    #   (If neurons/channels are not probabilistically independent from other
    #   neurons/channels, we may exploit this.)
    # Is "resolution" of outliers more important?


main()
