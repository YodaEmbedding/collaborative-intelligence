from typing import Any, Dict

import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras

from src.analysis import dataset, plot
from src.analysis.utils import basename_of, title_of


# TODO remove frequency y-label or scale it as a continuous density
# TODO do this over multiple dataset samples?
def analyze_histograms_layer(
    model_name: str,
    model_client: keras.Model,
    layer_name: str,
    layer_i: int,
    layer_n: int,
    take: int = 64,
    batch_size: int = 64,
) -> Dict[str, Any]:
    data = dataset.dataset().take(take).batch(batch_size)
    pred = model_client.predict(data)
    mean = np.mean(pred)
    std = np.std(pred)
    title = title_of(model_name, layer_name, layer_i, layer_n)
    basename = basename_of(model_name, layer_name, layer_i, layer_n)
    fig = plot.neuron_histogram(pred, title, bins=100)
    plot.save(fig, f"img/histogram/{basename}.png")
    print("{:8} {:8}".format("mean", "stddev"))
    print(f"{mean:8.2f} {std:8.2f}")
    return {"mean": mean, "std": std, "client_tensors": pred}
