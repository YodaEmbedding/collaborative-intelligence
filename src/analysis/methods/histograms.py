from typing import Any, Dict

import numpy as np
from tensorflow import keras

from src.analysis import plot
from src.analysis.dataset import single_sample_image
from src.analysis.utils import basename_of, title_of


# TODO remove frequency y-label or scale it as a continuous density
# TODO do this over multiple dataset samples?
def analyze_histograms_layer(
    model_name: str,
    model_client: keras.Model,
    layer_name: str,
    layer_i: int,
    layer_n: int,
) -> Dict[str, Any]:
    data = single_sample_image()[np.newaxis].astype(np.float32)
    pred = model_client.predict(data)
    mean = np.mean(pred)
    std = np.std(pred)
    title = title_of(model_name, layer_name, layer_i, layer_n)
    basename = basename_of(model_name, layer_name, layer_i, layer_n)
    fig = plot.neuron_histogram(pred, title, bins=100)
    plot.save(fig, f"img/histogram/{basename}.png")
    print("{:8} {:8}".format("mean", "stddev"))
    print(f"{mean:8.2f} {std:8.2f}")
    return {"mean": mean, "std": std}
