from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import src.analysis.dataset as ds
from src.analysis import plot
from src.analysis.experimentrunner import ExperimentRunner
from src.analysis.utils import tf_disable_eager_execution

tf_disable_eager_execution()

BATCH_SIZE = 64
DATASET_SIZE = 64


def featuremapsequence(
    frames: np.ndarray,
    tensors: np.ndarray,
    diffs: np.ndarray,
    title: str,
    order: str = "hwc",
    *,
    clim: Tuple[int, int] = None,
    cmap="viridis",
) -> plt.Figure:
    n = len(frames)
    ncols = 3
    nrows = n
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7, 10), constrained_layout=True
    )
    fill_value = clim[0] if clim is not None else None

    def plot_ax(ax, img):
        im = ax.matshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        if clim is not None:
            im.set_clim(*clim)

    for i, img in enumerate(frames):
        plot_ax(axes[i, 0], img)

    for i, arr in enumerate(tensors):
        img = plot.featuremap_image(arr, order, fill_value=fill_value)
        plot_ax(axes[i, 1], img)

    for i, arr in enumerate(diffs, start=1):
        img = plot.featuremap_image(arr, order, fill_value=fill_value)
        plot_ax(axes[i, 2], img)

    axes[0, 2].axis("off")

    # fig.suptitle(title, fontsize="xx-small")
    return fig


def main():
    runner = ExperimentRunner(
        model_name="resnet34",
        layer_name="add_3",
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )

    shape = runner.tensor_layout.shape
    dtype = runner.tensor_layout.dtype
    h, w, _c = shape

    dx_input_step = 224 // runner.tensor_layout.w
    dx_inputs = dx_input_step * np.arange(4) * 2
    dx_clients = np.arange(4) * 2
    dy_clients = np.zeros(4, dtype=np.int64)

    frames = ds.single_sample_image_xtrans(dx_inputs)
    client_tensors = runner.model_client.predict(frames)
    prev = client_tensors[0]
    diffs = np.empty((len(client_tensors) - 1, *shape), dtype=dtype)
    mses = np.empty(len(client_tensors) - 1)
    psnrs = np.empty(len(client_tensors) - 1)

    for i, curr in enumerate(client_tensors[1:]):
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        yy += dy_clients[i + 1]
        xx += dx_clients[i + 1]
        mask = (yy < 0) | (yy >= h) | (xx < 0) | (xx >= w)
        yy = np.clip(yy, 0, h - 1).reshape(-1)
        xx = np.clip(xx, 0, w - 1).reshape(-1)

        pred = prev[yy, xx].reshape(h, w, -1)
        diff = curr - pred
        diff[mask] = 0
        diffs[i] = diff

        x = diff[~mask]
        r = np.max(x) - np.min(x)
        mses[i] = np.mean(x ** 2)
        psnrs[i] = 20 * np.log(r ** 2 / mses[i])

    # axis = tuple(np.arange(1, diffs.ndim))
    # mses = np.mean(diffs**2, axis=axis)
    # r = np.max(diffs, axis=axis) - np.min(diffs, axis=axis)
    # psnrs = 20 * np.log(r**2 / mses)

    # print(diff[..., 0])
    print(mses)
    print(psnrs)

    # Adjust for visual purposes
    t = client_tensors
    off = t - np.mean(t)
    client_tensors = np.copysign(np.abs(off) ** 0.5, off) + np.mean(t)
    diffs = np.abs(diffs)

    suffix = "motionsequence"
    fig = featuremapsequence(frames, client_tensors, diffs, title="")
    plot.save(
        fig,
        f"img/experiment/{runner.basename}-{suffix}.png",
        bbox_inches="tight",
    )

    # TODO labels? "reference"? dx?

    # TODO subpixel motion compensation


main()
