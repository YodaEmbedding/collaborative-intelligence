import json
from io import BytesIO
from typing import ByteString

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.analysis import plot
from src.analysis.methods.histograms import analyze_histograms_layer
from src.analysis.methods.latencies import analyze_latencies_layer
from src.analysis.methods.motions import analyze_motions_layer
from src.analysis.utils import (
    compile_kwargs,
    get_cut_layers,
    release_models,
    separate_process,
)
from src.lib.split import split_model

# from src.analysis.accuracy_vs_kb import analyze_accuracy_vs_kb
# from src.analysis.dataset import dataset_kb

# On all models:
# 1. Output graph and text of model
# 2.
# 3.

# On all layers:
# 1. Output layer latencies
# 2. Output layer distribution histograms (=> generate models.json via stddev?)
# 3. Measure tensor optical flow on video (translate, rotation, scale, noise)

# On certain/recommended layers (that can be quantized):
# 1. Accuracy vs KB
# 2. Accuracy vs SNR/MSE
# 3.

# On ModelConfig layers:
# 1.
# 2.
# 3.

# Other:
# measure which has more accuracy: clip @ 90th pct, 99th pct, ...? (maybe make this an experiment!)
# mkdir -p img/graph img/summary
# stddev of stddevs?
# per-neuron stddev/mean/etc... or try to find structure in feature maps or whatever...


def analyze_layer(
    model_name: str, model: keras.Model, layer_name: str, i: int, n: int
):
    print(f"cut layer: {layer_name}")

    model_client, model_server, model_analysis = split_model(
        model, layer=layer_name
    )
    model_client.compile(**compile_kwargs)
    model_server.compile(**compile_kwargs)
    model_analysis.compile(**compile_kwargs)

    analyze_histograms_layer(model_name, model_client, layer_name, i, n)
    analyze_latencies_layer(model_client, layer_name)
    analyze_motions_layer(model_name, model_client, layer_name, i, n)

    release_models(model_client, model_server, model_analysis)
    print("")


# TODO memory: reload model for each separate task (or just comment out tasks)
# TODO optimization: reuse split models
@separate_process(sleep_after=5)
def analyze_model(model_name, cut_layers=None):
    print(f"Analyzing {model_name}...\n")
    prefix = f"models/{model_name}/{model_name}"
    model = keras.models.load_model(f"{prefix}-full.h5", compile=False)
    model.compile(**compile_kwargs)

    keras.utils.plot_model(model, to_file=f"img/graph/{model_name}.png")

    with open(f"img/summary/{model_name}.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(f"{x}\n"))

    if cut_layers is None:
        cut_layers = [x.name for x in get_cut_layers(model.layers[0])]

    n = len(cut_layers)

    for i, cut_layer_name in enumerate(cut_layers):
        analyze_layer(model_name, model, cut_layer_name, i, n)

    release_models(model)
    print("\n-----\n")


def load_model_from_bytestring(
    buf: ByteString, *args, **kwargs
) -> keras.Model:
    with BytesIO(buf) as f_buf:
        with h5py.File(f_buf, "r") as f_h5:
            model = keras.models.load_model(f_h5, *args, **kwargs)
    model.compile(**compile_kwargs)
    return model


def main():
    with open("models.json") as f:
        models = json.load(f)

    for model_name in models:
        analyze_model(model_name)


def main2():
    analyze_model("resnet34", cut_layers=["stage3_unit1_relu1", "add_7"])


def plot_test():
    basename = "test"
    frames = np.arange(60 * 72 * 72).reshape(60, 72, 72).astype(np.uint8)
    ani = plot.OpticalFlowAnimator(frames, frames, frames, f"{basename}")
    ani.save_img(f"img/motion/{basename}.png")
    ani.save(f"img/motion/{basename}.mp4")


if __name__ == "__main__":
    # plot_test()
    # main()
    main2()
