from contextlib import suppress
from math import ceil, sqrt
from typing import Dict, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

from src.lib.layouts import TensorLayout
from src.lib.tile import determine_tile_layout, tile


def featuremap_image(
    arr: np.ndarray, order: str = "hwc", fill_value=None
) -> np.ndarray:
    if fill_value is None:
        fill_value = np.min(arr)
    tensor_layout = TensorLayout.from_tensor(arr, order)
    tiled_layout = determine_tile_layout(tensor_layout)
    tiled = tile(arr, tensor_layout, tiled_layout, fill_value=fill_value)
    return tiled


def featuremap(
    arr: np.ndarray,
    title: str,
    order: str = "hwc",
    *,
    cbar: bool = True,
    clim: Tuple[int, int] = None,
    cmap="viridis",
) -> plt.Figure:
    if arr.ndim != 2:
        fill_value = clim[0] if clim is not None else None
        img = featuremap_image(arr, order, fill_value=fill_value)
    else:
        img = arr
    fig, ax = plt.subplots(tight_layout=True)
    im = ax.matshow(img, cmap=cmap)
    ax.set_title(title, fontsize="xx-small")
    ax.set_xticks([])
    ax.set_yticks([])
    if clim is not None:
        im.set_clim(*clim)
    if cbar:
        fig.colorbar(im)
    return fig


def featuremapcompression(
    samples: Dict[float, np.ndarray],
    title: str,
    order: str = "hwc",
    *,
    clim: Tuple[int, int] = None,
    cmap="viridis",
) -> plt.Figure:
    n = len(samples)
    ncols = int(ceil(sqrt(len(samples))))
    nrows = int(ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8, 9), constrained_layout=True
    )
    fill_value = clim[0] if clim is not None else None
    for i, (kb, arr) in enumerate(samples.items()):
        img = featuremap_image(arr, order, fill_value=fill_value)
        ax = axes[i // ncols, i % ncols] if nrows > 1 else axes[i]
        im = ax.matshow(img, cmap=cmap)
        ax.set_title(f"{kb:.0f} KB", y=-0.07)
        ax.set_xticks([])
        ax.set_yticks([])
        if clim is not None:
            im.set_clim(*clim)
    for i in range(n, nrows * ncols):
        ax = axes[i // ncols, i % ncols] if nrows > 1 else axes[i]
        ax.axis("off")
    fig.suptitle(title, fontsize="xx-small")
    return fig


def model_bar(heights, xlabels, title: str, ylabel: str) -> plt.Figure:
    x = np.arange(len(heights))
    fig, ax = plt.subplots()
    rect = ax.bar(x, heights)
    n = len(xlabels)
    font_scale_factor = 320
    fontsize = min(10, int(2 * font_scale_factor / n) / 2)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=fontsize, rotation=-90)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(bottom=0.35)
    return fig


def neuron_histogram(
    arr: np.ndarray,
    title: str,
    bins: int = 20,
    clip_range: Tuple[int, int] = None,
) -> plt.Figure:
    arr = arr.reshape((-1,))
    if clip_range is None:
        clip_range = (np.min(arr), np.max(arr))
    bins_ = np.linspace(*clip_range, bins)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(arr, bins=bins_, density=True)
    ax.set_xlabel("Neuron value")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize="xx-small")
    return fig


def save(fig: plt.Figure, filename: str, close=True, dpi=300, **kwargs):
    fig.savefig(filename, dpi=dpi, **kwargs)
    if close:
        fig.clf()
        plt.close(fig)


def colormap_upper(
    levels: int,
    *,
    cmin: float = 0.0,
    cmax: float = 1.0,
    cmap: str = "viridis",
    gamma: float = 2.0,
) -> ListedColormap:
    """Colormap with higher resolution around upper end of range"""
    crange = np.linspace(cmin, cmax, levels)
    colormap = cm.get_cmap(cmap, 100000)
    return ListedColormap(colormap(crange ** gamma))


class OpticalFlowAnimator:
    def __init__(self, frames, tensors, flows, title):
        if len(frames) != len(tensors) or len(frames) != len(flows):
            raise ValueError("Mismatching number of frames, tensors, flows")
        self.frames = frames
        self.tensors = tensors
        self.flows = flows
        # self.fig = plt.figure()
        self.fig = plt.figure(figsize=(8, 8 / 3 * 1.1))

        # w = 6
        # self.fig = plt.figure(figsize=(w, w / 3 + 0.5))
        # self.fig.tight_layout()

        self.ax_fram = self.fig.add_subplot(1, 3, 1)
        self.ax_tens = self.fig.add_subplot(1, 3, 2)
        self.ax_flow = self.fig.add_subplot(1, 3, 3)

        # self.fig, self.axes = plt.subplots(1, 3, squeeze=True)
        # self.fig, self.axes = plt.subplots(1, 3)
        # self.ax_fram, self.ax_tens, self.ax_flow = list(self.axes)
        # self.ax_fram = self.axes[0]
        # self.ax_tens = self.axes[1]
        # self.ax_flow = self.axes[2]

        self.im_fram = self.ax_fram.imshow(
            self.frames[0], interpolation="nearest"
        )

        self.im_tens = self.ax_tens.matshow(
            self.tensors[0], interpolation="nearest", cmap="viridis"
        )
        self.im_tens.set_clim(np.min(self.tensors), np.max(self.tensors))
        # self.fig.colorbar(self.im_tens)

        self.y_flow, self.x_flow = np.meshgrid(
            np.arange(0, self.flows.shape[1]),
            np.arange(0, self.flows.shape[2]),
        )

        self.im_flow = self.ax_flow.quiver(
            self.x_flow,
            self.y_flow,
            self.flows[0, ..., 0],
            self.flows[0, ..., 1],
        )
        self.im_flow.set_clim(np.min(self.flows), np.max(self.flows))

        # self.im_flow = self.ax_flow.matshow(
        #     self.flows[0], interpolation="nearest", cmap="viridis"
        # )
        # self.im_flow.set_clim(np.min(self.flows), np.max(self.flows))
        # self.fig.colorbar(self.im_flow)

        self.fig.suptitle(title)
        # self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        # self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.fig.tight_layout()

        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=len(self.flows)
        )

    def __del__(self):
        with suppress(AttributeError):
            plt.close(self.fig)

    def animate(self, i):
        self.im_fram.set_data(self.frames[i])
        self.im_tens.set_data(self.tensors[i])
        self.im_flow.set_UVC(self.flows[i, ..., 0], self.flows[i, ..., 1])
        self.ax_fram.set_xticks([])
        self.ax_fram.set_yticks([])
        self.ax_tens.set_xticks([])
        self.ax_tens.set_yticks([])
        self.ax_flow.set_xticks([])
        self.ax_flow.set_yticks([])

    def save(self, filename, dpi=1200, **kwargs):
        writer = animation.writers["ffmpeg"](fps=5, bitrate=2000)
        self.ani.save(filename, dpi=dpi, writer=writer, **kwargs)
        plt.close(self.fig)

    def save_img(self, filename, frame_num=0, dpi=200, **kwargs):
        self.animate(frame_num)
        self.fig.savefig(filename, dpi=dpi, **kwargs)
        plt.close(self.fig)


class MatshowAnimator:
    def __init__(self, frames, title):
        self.frames = frames
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_title(title)
        self.im = self.ax.matshow(self.frames[0], cmap="viridis")
        self.im.set_clim(np.min(self.frames), np.max(self.frames))
        self.fig.colorbar(self.im)
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=len(self.frames)
        )

    def __del__(self):
        self.fig.clf()
        plt.close(self.fig)

    def animate(self, i):
        z = self.frames[i]
        self.im.set_data(z)
        # self.im.set_clim(np.min(z), np.max(z))
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def save(self, filename, dpi=200, **kwargs):
        writer = animation.writers["ffmpeg"](fps=5, bitrate=2000)
        self.ani.save(filename, dpi=dpi, writer=writer, **kwargs)

    def save_img(self, filename, frame_num=0, dpi=200, **kwargs):
        self.animate(frame_num)
        self.fig.savefig(filename, dpi=dpi, **kwargs)
