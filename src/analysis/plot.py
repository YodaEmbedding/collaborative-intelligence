import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K  # pylint: disable=import-error

from src.tile import TensorLayout, determine_tile_layout, tile


def featuremap(arr: np.ndarray, title: str, order: str = "hwc") -> plt.Figure:
    tensor_layout = TensorLayout.from_tensor(arr, order)
    tiled_layout = determine_tile_layout(tensor_layout)
    tiled = tile(arr, tensor_layout, tiled_layout)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(tiled, cmap="viridis")
    ax.set_title(title, fontsize="xx-small")
    fig.colorbar(cax)
    return fig


def neuron_histogram(arr: np.ndarray, title: str) -> plt.Figure:
    arr = arr.reshape((-1,))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(arr, bins=np.linspace(np.min(arr), np.max(arr), 20))
    ax.set_xlabel("Neuron value")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize="xx-small")
    return fig
