import json

import numpy as np
import skimage.measure
import tensorflow as tf

from src.analysis import dataset as ds
from src.analysis import plot
from src.analysis.experimentrunner import ExperimentRunner
from src.analysis.quant import uni_dequant, uni_quant
from src.analysis.utils import (
    categorical_top1_accuracy,
    tf_disable_eager_execution,
)
from src.lib.layouts import TensorLayout
from src.lib.postencode import _pil_encode
from src.lib.predecode import _decode_raw_img
from src.lib.tile import determine_tile_layout, tile

tf_disable_eager_execution()

with open("config.json") as f:
    config = json.load(f)

BATCH_SIZE = config["batch_size"]
DATASET_SIZE = config["dataset_size"]
TEST_DATASET_SIZE = config["test_dataset_size"]


def zigzag(h, w):
    zz = np.empty(h * w, dtype=np.int64)
    x, y = 0, 0
    state = 0
    for i in range(h * w):
        zz[i] = y * w + x
        if state == 0:
            if x < w - 1:
                x += 1
            else:
                y += 1
            state += 1
        elif state == 1:
            x -= 1
            y += 1
            if x == 0 or y == h - 1:
                state += 1
        elif state == 2:
            if y < h - 1:
                y += 1
            else:
                x += 1
            state += 1
        elif state == 3:
            x += 1
            y -= 1
            if x == w - 1 or y == 0:
                state = 0
    return zz


def zigzag_horz(h, w):
    zz = np.arange(h * w).reshape((h, w))
    zz[1::2] = zz[1::2][:, ::-1]
    return zz.reshape(-1)


def main():
    # runner = ExperimentRunner(
    #     model_name="resnet34",
    #     layer_name="add_3",
    #     dataset_size=1,
    #     batch_size=1,
    #     test_dataset_size=1,
    #     # dataset_size=DATASET_SIZE,
    #     # batch_size=BATCH_SIZE,
    #     # test_dataset_size=TEST_DATASET_SIZE,
    # )
    # tensor_layout = runner.tensor_layout

    model_name = "resnet34"
    basename = f"{model_name}-16of37-add_3"
    title = f"{model_name} add_3 (16/37)"
    basename_stats = f"img/stats/{model_name}/{basename}"

    try:
        img = ds.single_sample_image()
        x_client = runner.model_client.predict(img[np.newaxis])[0]
        np.save(f"img/experiment/{basename}-tensor.npy", x_client)
    except NameError:
        x_client = np.load(f"img/experiment/{basename}-tensor.npy")
        tensor_layout = TensorLayout.from_tensor(x_client, "hwc")

    def plot_featuremap(arr: np.ndarray, suffix: str):
        filename = f"img/experiment/{basename}-{suffix}.png"
        fig = plot.featuremap(arr, title)
        plot.save(fig, filename, bbox_inches="tight")

    tensors_mean = np.load(f"{basename_stats}-tensors_mean.npy")
    h, w, c = tensor_layout.shape

    plot_featuremap(x_client, "tensor_client")

    mean = np.mean(x_client)
    std = np.std(x_client)
    x = x_client
    # x[:, 0] -= 2 * std
    # x[0, :] -= 2 * std
    # x[:, w - 1] -= 2 * std
    # x[h - 1, :] -= 2 * std
    x = np.clip(x, mean - 3 * std, mean + 3 * std)
    x_clip = x

    plot_featuremap(x_clip, "tensor_client_clip")
    plot_featuremap(x_clip, "tensor_client_clip")

    # TODO load and normalize by proper means/std because otherwise things go weird?

    x_norm = (x_client - np.mean(x_client, axis=(0, 1))) / np.std(
        x_client, axis=(0, 1)
    )
    plot_featuremap(np.clip(x_norm, -3, 3), "tensor_norm")

    sim_mat = np.zeros((c, c))
    bw = 7
    for c1 in range(c):
        for c2 in range(c):
            x1 = x_norm[..., c1]
            x2 = x_norm[..., c2]
            x1 = skimage.measure.block_reduce(np.abs(x1), (bw, bw), np.max)
            x2 = skimage.measure.block_reduce(np.abs(x2), (bw, bw), np.max)
            s = np.mean((x1 - x2) ** 2)
            sim_mat[c1, c2] = s
    sim_mat_nodiag = sim_mat.copy()
    sim_mat_nodiag[np.diag_indices(c)] = np.inf
    sim_min = np.min(sim_mat_nodiag, axis=1)[np.newaxis, np.newaxis]
    sim_min_featmap = np.broadcast_to(sim_min, (h, w, c))
    sim_argmin = np.argmin(sim_mat_nodiag, axis=1)[np.newaxis, np.newaxis]
    sim_argmin_featmap = np.broadcast_to(sim_argmin, (h, w, c))
    plot_featuremap(sim_mat, "tensor_similarity")
    plot_featuremap(sim_min_featmap, "tensor_similaritymin")
    plot_featuremap(sim_argmin_featmap, "tensor_similarityargmin")
    k = 16
    r_sim = 3
    with np.printoptions(precision=1, floatmode="fixed", suppress=True):
        # print(sim_mat[:k, :k])
        # print(10 * np.log10(r_sim ** 2 / sim_mat[:k, :k]))
        print(sim_min)
        print(10 * np.log10(r_sim ** 2 / sim_min))

    # tiled_layout = determine_tile_layout(tensor_layout)
    # x = tile(x, tensor_layout, tiled_layout)

    x = x.reshape(-1, c)

    zz = zigzag(h, w)
    x = x[zz]

    # zz = zigzag_horz(h, w)
    # x = x[zz]

    mu = np.mean(x_client.reshape(-1, c), axis=0)
    idx = np.argsort(mu)
    x = x[:, idx]

    x_orig = x

    levels = 256
    clip_range = (-3 * std, 3 * std)
    x = x_orig
    x = uni_quant(x, clip_range, levels)
    x = _pil_encode(x, "JPEG", quality=3)
    b = len(x)
    x = np.array(_decode_raw_img(x))
    x = uni_dequant(x, clip_range, levels)
    x_recv = x

    r = np.max(x_client) - np.min(x_client)
    err = x_recv - x_orig
    mse = np.mean((err) ** 2)
    psnr = 10 * np.log(r ** 2 / mse)
    print(f"MSE:  {mse:.4f}")
    print(f"PSNR: {psnr:.1f} dB")
    print(f"Size: {b} bytes")

    plot_featuremap(x_recv, "tensor_recv")

    # x = runner.model_server.predict(x_client[np.newaxis])[0]

    # TODO crossentropy
    # TODO try difference instead
    # TODO consider changing stdev of image that goes into the codec itself...!
    # TODO "mu" should be precomputed on dataset
    # TODO JPEG2000/wavelet compression on quilted tensor

    # - Maybe try "similarity matrix" with finer regions than channels
    # - In channels that look like noise, is high frequency content important?
    #   Try "disabling" those channels (e.g. with the actual_mean or tensors_mean)

    # Save statistical properties in npy or pandas files...


main()
