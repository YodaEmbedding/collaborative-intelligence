from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

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
    preds: np.ndarray,
    diffs: np.ndarray,
    title: str,
    order: str = "hwc",
    *,
    clim: Tuple[int, int] = None,
    clim_diff: Tuple[int, int] = None,
    cmap="viridis",
) -> plt.Figure:
    n = len(frames)
    ncols = 4
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.3, 10))
    fig.subplots_adjust(wspace=0.0, hspace=0.1)
    fill_value = clim[0] if clim is not None else None

    def plot_ax(ax, img, *, clim=clim, cbar=False):
        im = ax.matshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        if clim is not None:
            im.set_clim(*clim)
        if cbar:
            cax = fig.add_axes([
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.01,
                ax.get_position().height,
            ])
            cax.tick_params(labelsize=8)
            fig.colorbar(im, cax=cax)

    for i, img in enumerate(frames):
        plot_ax(axes[i, 0], img)

    for i, arr in enumerate(tensors):
        img = plot.featuremap_image(arr, order, fill_value=fill_value)
        plot_ax(axes[i, 1], img)

    for i, arr in enumerate(preds, start=1):
        img = plot.featuremap_image(arr, order, fill_value=fill_value)
        plot_ax(axes[i, 2], img)

    for i, arr in enumerate(diffs, start=1):
        img = plot.featuremap_image(arr, order, fill_value=fill_value)
        plot_ax(axes[i, 3], img, cbar=True, clim=clim_diff)

    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

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
    h, w, c = shape

    n = 4
    px = 2
    offset = 0.25
    x_in_per_cl = 224 / w
    dx_inputs = (((np.arange(n) * px) + offset) * x_in_per_cl).astype(np.int64)
    dx_inputs[0] = 0
    dx_clients = dx_inputs / x_in_per_cl
    dy_clients = np.zeros(n)

    frames = ds.single_sample_image_xtrans(dx_inputs)
    client_tensors = runner.model_client.predict(frames)
    prev = client_tensors[0]
    diffs = np.empty((len(client_tensors) - 1, *shape), dtype=dtype)
    preds = np.empty((len(client_tensors) - 1, *shape), dtype=dtype)
    mses = np.empty(len(client_tensors) - 1)
    psnrs = np.empty(len(client_tensors) - 1)

    for i, curr in enumerate(client_tensors[1:]):
        r = np.max(curr) - np.min(curr)
        yy, xx, cc = np.meshgrid(
            np.arange(h), np.arange(w), np.arange(c), indexing="ij"
        )
        yy = yy + dy_clients[i + 1]
        xx = xx + dx_clients[i + 1]
        mask = (yy < 0) | (yy > h - 1) | (xx < 0) | (xx > w - 1)
        yy = np.clip(yy, 0, h - 1).reshape(-1)
        xx = np.clip(xx, 0, w - 1).reshape(-1)
        cc = cc.reshape(-1)

        pred = ndimage.map_coordinates(prev, [yy, xx, cc], order=3)
        pred = pred.reshape(h, w, -1)
        preds[i] = pred

        diff = curr - pred
        diff[mask] = 0
        diffs[i] = diff

        x = diff[~mask]
        mses[i] = np.mean(x ** 2)
        psnrs[i] = 10 * np.log(r ** 2 / mses[i])

        # preds[mask] = 0

    print(dx_inputs)
    print(dx_clients)
    print(mses)
    print(psnrs)

    def visual_adjust_tensor(t):
        off = t - np.mean(t)
        return np.copysign(np.abs(off) ** 0.5, off) + np.mean(t)

    # Adjust for visual purposes
    tensors_ = visual_adjust_tensor(client_tensors)
    preds_ = visual_adjust_tensor(preds)
    diffs_ = np.abs(diffs)

    # Show only k channels
    koff = 0
    k = 4 ** 2
    tensors_ = tensors_[..., koff : koff + k]
    preds_ = preds_[..., koff : koff + k]
    diffs_ = diffs_[..., koff : koff + k]

    # Scale colorbar in a consistent manner
    clim = (tensors_.min(), tensors_.max())
    r = clim[1] - clim[0]
    clim_diff = (0, r)

    # Manual overrides
    clim = (-1.6, 1.4)
    clim_diff = (0, 3.0)

    fig = featuremapsequence(
        frames,
        tensors_,
        preds_,
        diffs_,
        title="",
        clim=clim,
        clim_diff=clim_diff,
    )

    def idx(y, x):
        return 4 * y + x

    pad = 6
    ax_opts = dict(
        xy=(0.5, 1),
        xycoords="axes fraction",
        xytext=(0, pad),
        textcoords="offset points",
        size="medium",
        ha="center",
        va="baseline",
    )
    axes = fig.axes
    axes[idx(0, 0)].annotate("Image", **ax_opts)
    axes[idx(0, 1)].annotate("Tensor", **ax_opts)
    axes[idx(1, 2)].annotate("Motion compensated\ntensor", **ax_opts)
    axes[idx(1, 3)].annotate("Difference", **ax_opts)
    axes[idx(0, 0)].set_ylabel("Reference")
    axes[idx(1, 0)].set_ylabel(f"{dx_inputs[1]} px")
    axes[idx(2, 0)].set_ylabel(f"{dx_inputs[2]} px")
    axes[idx(3, 0)].set_ylabel(f"{dx_inputs[3]} px")

    suffix = f"input_translate_{px}px_{offset}plus_order3"
    plot.save(
        fig,
        f"img/experiment/{runner.basename}-{suffix}.png",
        bbox_inches="tight",
    )


main()
