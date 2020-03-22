import json
from io import BytesIO
from typing import ByteString

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.analysis import plot

# from src.analysis.accuracy_vs_kb import analyze_accuracy_vs_kb
# from src.analysis.dataset import dataset_kb
from src.analysis.methods.histograms import analyze_histograms_layer
from src.analysis.methods.latencies import analyze_latencies_layer
from src.analysis.methods.motions import analyze_motions_layer
from src.analysis.utils import (
    compile_kwargs,
    get_cut_layers,
    release_models,
    separate_process,
)

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


# TODO plot bar chart
# TODO exclude GPU -> RAM data copy time (e.g. via Stage or tf.profiler)
# TODO plot stacked bar chart including GPU -> RAM data copy time
def analyze_latencies(model: keras.Model, layers: List[str]):
    print("{:20}{}".format("layer", "cumulative time (ms)"))
    for split_layer in layers:
        analyze_latencies_layer(model, split_layer)


# TODO remove frequency y-label or scale it as a continuous density
# TODO do this over multiple dataset samples?
def analyze_histograms(model_name: str, model: keras.Model, layers: List[str]):
    n = len(layers)
    print("{:20}{:8}{:8}".format("layer", "mean", "stddev"))
    for i, split_layer in enumerate(layers):
        analyze_histograms_layer(model_name, model, split_layer, i, n)


def analyze_motions(model_name: str, model: keras.Model, layers: List[str]):
    n = len(layers)
    print("{:20}".format("layer"))
    for i, split_layer in enumerate(layers):
        analyze_motions_layer(model_name, model, split_layer, i, n)


# TODO memory: reload model for each separate task (or just comment out tasks)
# TODO optimization: reuse split models
@separate_process(sleep_after=5)
def analyze_model(model_name):
    print(f"Analyzing {model_name}...\n")
    prefix = f"models/{model_name}/{model_name}"
    model = keras.models.load_model(f"{prefix}-full.h5", compile=False)
    model.compile(**compile_kwargs)
    # with open(f"{prefix}-full.h5", "rb") as f:
    #     h5_data = f.read()
    # model = load_model_from_bytestring(h5_data, compile=False)

    keras.utils.plot_model(model, to_file=f"img/graph/{model_name}.png")

    with open(f"img/summary/{model_name}.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(f"{x}\n"))

    cut_layers = [x.name for x in get_cut_layers(model.layers[0])]

    analyze_latencies(model, cut_layers)
    analyze_histograms(model_name, model, cut_layers)
    analyze_motions(model_name, model, cut_layers)

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

    # models = {
    #     model: [ModelConfig(model=model, **x) for x in opt_list]
    #     for model, opt_list in models.items()
    # }

    for model_name in models:
        analyze_model(model_name)


def main2():
    model_name = "resnet34"
    print(f"Analyzing {model_name}...\n")
    prefix = f"models/{model_name}/{model_name}"
    model = keras.models.load_model(f"{prefix}-full.h5", compile=False)
    model.compile(**compile_kwargs)

    keras.utils.plot_model(model, to_file=f"img/graph/{model_name}.png")

    with open(f"img/summary/{model_name}.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(f"{x}\n"))

    # TODO this includes input and output layers... which ones are splittable?
    # cut_layers = list(get_cut_layers(model.layers[0]))
    cut_layers = ["stage3_unit1_relu1", "add_7"]

    # analyze_latencies(model, cut_layers)
    # analyze_histograms(model_name, model, cut_layers)
    analyze_motions(model_name, model, cut_layers)

    release_models(model)
    print("\n-----\n")


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
