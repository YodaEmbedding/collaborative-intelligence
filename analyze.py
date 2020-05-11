import json
from io import BytesIO
from typing import ByteString

import h5py
import tensorflow as tf
from tensorflow import keras

from src.analysis.experimentrunner import ExperimentRunner
from src.analysis.methods.accuracyvskb import analyze_accuracyvskb_layer
from src.analysis.methods.featuremap import (
    analyze_featuremap_layer,
    analyze_featuremapcompression_layer,
)
from src.analysis.methods.histograms import analyze_histograms_layer
from src.analysis.methods.latencies import (
    analyze_latencies_layer,
    analyze_latencies_post,
)
from src.analysis.methods.motions import analyze_motions_layer
from src.analysis.methods.size import analyze_size_model
from src.analysis.methods.stats import analyze_stats_layer
from src.analysis.quant import uni_dequant, uni_quant
from src.analysis.utils import (
    basename_of,
    compile_kwargs,
    dataset_to_numpy_array,
    get_cut_layers,
    new_tf_graph_and_session,
    release_models,
    separate_process,
    tf_disable_eager_execution,
    title_of,
)
from src.lib.split import split_model

tf_disable_eager_execution()

BATCH_SIZE = 64
DATASET_SIZE = 4096


def analyze_layer(runner: ExperimentRunner):
    model_name = runner.model_name
    layer_name = runner.layer_name
    model = runner.model
    model_client = runner.model_client
    model_server = runner.model_server
    title = runner.title
    basename = runner.basename

    print(title)

    analyze_stats_layer(runner)

    d = {"layer": layer_name}

    if model_client.input == model_client.output:
        model_client.predict = dataset_to_numpy_array
        model_client.predict_on_batch = dataset_to_numpy_array

    d.update(analyze_histograms_layer(model_client, title, basename))

    analyze_featuremap_layer(model_client, title, basename)

    clip_range = (d["mean"] - 4 * d["std"], d["mean"] + 4 * d["std"])
    quant = lambda x: uni_quant(x, clip_range=clip_range, levels=256)
    kbs = [2, 5, 10, 30]
    analyze_featuremapcompression_layer(
        model_client, title, basename, quant, kbs=kbs
    )

    analyze_motions_layer(model_client, title, basename)

    args = (model_name, model, model_client, model_server, title, basename)
    clip_range = (d["mean"] - 3 * d["std"], d["mean"] + 3 * d["std"])
    quant = lambda x: uni_quant(x, clip_range=clip_range, levels=256)
    dequant = lambda x: uni_dequant(x, clip_range=clip_range, levels=256)
    postencoder_name = "jpeg"
    subdir = f"{postencoder_name}_uniquant256/{model_name}"
    analyze_accuracyvskb_layer(
        *args, quant, dequant, postencoder_name, BATCH_SIZE, subdir
    )

    d["latency"] = analyze_latencies_layer(model_client, layer_name)

    runner.close()
    print("")

    return d


# TODO memory: reload model for each separate task (or just comment out tasks)
# TODO optimization: reuse split models
@separate_process(sleep_after=5)
@new_tf_graph_and_session
def load_model_and_run(model_name, func):
    prefix = f"models/{model_name}/{model_name}"
    model = keras.models.load_model(f"{prefix}-full.h5", compile=False)
    model.compile(**compile_kwargs)
    result = func(model)
    release_models(model)
    return result


@separate_process(sleep_after=5)
@new_tf_graph_and_session
def tf_run_isolated(func, *args, **kwargs):
    return func(*args, **kwargs)


def analyze_model(model_name, layers=None):
    print(f"Analyzing {model_name}...\n")

    def init(model):
        keras.utils.plot_model(model, to_file=f"img/graph/{model_name}.png")
        with open(f"img/summary/{model_name}.txt", "w") as f:
            model.summary(print_fn=lambda x: f.write(f"{x}\n"))
        cut_layers = [x.name for x in get_cut_layers(model.layers[0])]
        analyze_size_model(model_name, model, cut_layers)
        return cut_layers

    cut_layers = load_model_and_run(model_name, init)
    dicts = []

    if layers is None:
        layers = cut_layers

    def analyze_layer_wrapper(cut_layer_name):
        runner = ExperimentRunner(
            model_name,
            cut_layer_name,
            dataset_size=DATASET_SIZE,
            batch_size=BATCH_SIZE,
        )
        return analyze_layer(runner)

    for cut_layer_name in cut_layers:
        if cut_layer_name not in layers:
            dicts.append({})
            continue

        d = tf_run_isolated(analyze_layer_wrapper, cut_layer_name)
        dicts.append(d)

    analyze_latencies_post(model_name, dicts)

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
    global analyze_featuremap_layer
    global analyze_featuremapcompression_layer
    global analyze_latencies_post
    global analyze_motions_layer
    global analyze_size_model

    identity = lambda *args, **kwargs: None

    analyze_size_model = identity
    analyze_latencies_post = identity
    analyze_featuremap_layer = identity
    analyze_featuremapcompression_layer = identity
    analyze_motions_layer = identity

    # layers = ["pooling0", "add_3", "add_7", "add_13"]
    # layers = (
    #     [f"add_{i}" for i in range(1, 16)]
    #     + ["add"]
    #     + [f"stage{i}_unit1_bn1" for i in range(1, 5)]
    #     + [f"stage{i}_unit1_relu1" for i in range(1, 5)]
    # )
    # analyze_model("resnet34", layers=layers)
    analyze_model("resnet34")


if __name__ == "__main__":
    main()
