import numpy as np
from tensorflow import keras

from src.analysis import plot
from src.analysis.dataset import single_sample_image
from src.analysis.utils import (
    basename_of,
    compile_kwargs,
    release_models,
    title_of,
)
from src.lib.split import split_model


# @separate_process()
def analyze_histograms_layer(
    model_name: str,
    model: keras.Model,
    layer_name: str,
    layer_i: int,
    layer_n: int,
):
    model_client, model_server, model_analysis = split_model(
        model, layer=layer_name
    )
    model_client.compile(**compile_kwargs)
    model_server.compile(**compile_kwargs)
    model_analysis.compile(**compile_kwargs)
    data = single_sample_image()[np.newaxis].astype(np.float32)
    pred = model_client.predict(data)
    mean = np.mean(pred)
    std = np.std(pred)
    title = title_of(model_name, layer_name, layer_i, layer_n)
    basename = basename_of(model_name, layer_name, layer_i, layer_n)
    fig = plot.neuron_histogram(pred, title, bins=100)
    plot.save(fig, f"img/histogram/{basename}.png")
    release_models(model_client, model_server, model_analysis)
    print(f"{layer_name:20}{mean:8.2f}{std:8.2f}")
