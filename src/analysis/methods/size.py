from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

from src.analysis import plot


def analyze_size_model(
    model_name: str, model: keras.Model, layer_names: List[str]
):
    """Plot histogram of tensor sizes for given layers."""
    title = f"{model_name}"
    basename = f"{model_name}"
    xlabels = layer_names
    layers = [model.get_layer(x) for x in layer_names]
    heights = [layer_output_size(x)[0] / 1024 for x in layers]
    fig = plot.model_bar(heights, xlabels, title, "Output tensor size (KiB)")
    plot.save(fig, f"img/sizes/{basename}.png")


def layer_output_size(layer: Layer) -> Tuple[int, int]:
    """Returns total size in bytes and total size in number of neurons."""
    bpn_lut = {tf.uint8: 1, "uint8": 1, tf.float32: 4, "float32": 4}
    bpn = bpn_lut[layer.dtype]
    output_shape = layer.output_shape
    if isinstance(output_shape, list):
        if len(output_shape) > 1:
            raise Exception("More than one output_shape found")
        output_shape = output_shape[0]
    n = np.prod(list(filter(None, output_shape)))
    return bpn * n, n
