from contextlib import suppress

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.lib.layouts import TensorLayout
from src.lib.tile import determine_tile_layout, tile


def featuremap(arr: np.ndarray, title: str, order: str = "hwc") -> plt.Figure:
    tensor_layout = TensorLayout.from_tensor(arr, order)
    tiled_layout = determine_tile_layout(tensor_layout)
    tiled = tile(arr, tensor_layout, tiled_layout)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.matshow(tiled, cmap="viridis")
    ax.set_title(title, fontsize="xx-small")
    fig.colorbar(im)
    return fig


def neuron_histogram(
    arr: np.ndarray, title: str, bins: int = 20
) -> plt.Figure:
    arr = arr.reshape((-1,))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(arr, bins=np.linspace(np.min(arr), np.max(arr), bins))
    ax.set_xlabel("Neuron value")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize="xx-small")
    return fig


def save(fig: plt.Figure, filename: str, close: bool = True):
    fig.savefig(filename, dpi=200)
    if close:
        fig.clf()
        plt.close(fig)


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
