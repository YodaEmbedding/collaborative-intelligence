from typing import Callable, List

import numpy as np
from tensorflow import keras

from src.analysis import plot
from src.analysis.dataset import single_sample_image
from src.analysis.utils import basename_of, title_of
from src.lib.layouts import TensorLayout
from src.lib.postencode import JpegPostencoder
from src.lib.predecode import JpegPredecoder


def analyze_featuremap_layer(
    model_name: str,
    model_client: keras.Model,
    layer_name: str,
    layer_i: int,
    layer_n: int,
):
    shape = model_client.output_shape[1:]
    if len(shape) != 3:
        return

    data = single_sample_image()[np.newaxis].astype(np.float32)
    tensor = model_client.predict(data)[0]

    title = title_of(model_name, layer_name, layer_i, layer_n)
    basename = basename_of(model_name, layer_name, layer_i, layer_n)
    fig = plot.featuremap(tensor, title, cbar=False)
    plot.save(fig, f"img/featuremap/{basename}.png")

    print("Analyzed featuremap")


def analyze_featuremapcompression_layer(
    model_name: str,
    model_client: keras.Model,
    layer_name: str,
    layer_i: int,
    layer_n: int,
    quant: Callable[[np.ndarray], np.ndarray],
    kbs: List[float],
):
    shape = model_client.output_shape[1:]
    if len(shape) != 3:
        return
    # TODO assert dtype?
    dtype = model_client.dtype

    data = single_sample_image()[np.newaxis].astype(np.float32)
    tensor = quant(model_client.predict(data)[0])
    tensor_layout = TensorLayout.from_shape(shape, "hwc", dtype)
    tiled_layout = JpegPostencoder(tensor_layout, quality=100).tiled_layout
    predecoder = JpegPredecoder(tiled_layout, tensor_layout)

    samples = {}

    for quality in range(1, 101):
        postencoder = JpegPostencoder(tensor_layout, quality=quality)
        buf = postencoder.run(tensor)
        kbb = len(buf) / 1024
        kb = min(kbs, key=lambda x, y=kbb: abs(y - x))
        if kb in samples and abs(len(samples[kb]) / 1024 - kb) < abs(kbb - kb):
            continue
        samples[kb] = buf

    samples = {
        round(len(x) / 1024): predecoder.run(x) for _, x in samples.items()
    }
    title = title_of(model_name, layer_name, layer_i, layer_n)
    basename = basename_of(model_name, layer_name, layer_i, layer_n)
    fig = plot.featuremapcompression(samples, title, clim=(0, 255))
    plot.save(fig, f"img/featuremapcompression/{basename}.png")

    print("Analyzed featuremapcompression")
