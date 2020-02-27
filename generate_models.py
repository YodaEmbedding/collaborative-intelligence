import gc
import json
import os
import textwrap
from contextlib import suppress
from pprint import pprint
from typing import Iterator

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # pylint: disable=import-error
from classification_models.tfkeras import Classifiers
from tensorflow import keras
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model

from src.analysis import plot
from src.analysis.video import read_video, write_video
from src.lib.layouts import TensorLayout, TiledArrayLayout
from src.lib.split import copy_model
from src.lib.tile import determine_tile_layout
from src.modelconfig import ModelConfig
from src.utils import split_model_by_config

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def convert_to_tflite_model(model: keras.Model, tflite_filename: str):
    """Convert keras model file to tflite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)


def create_model(model_name: str) -> keras.Model:
    """Model factory."""
    shape = (224, 224, 3)
    model_creator, _ = Classifiers.get(model_name)
    return model_creator(shape, weights="imagenet")


def prefix_of(model_config: ModelConfig) -> str:
    return f"models/{model_config.to_path()}"


def prefix_of_name(model_name: str) -> str:
    return f"models/{model_name}/{model_name}"


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    return -(np.sum(targets * np.log(predictions))) / predictions.shape[0]


def get_preprocessor(model_name: str):
    """Get input preprocessor for model."""
    return Classifiers.get(model_name)[1]


def single_input_image(filename: str, target_size=(224, 224)) -> np.ndarray:
    """Load single image for testing."""
    img = image.load_img(filename, target_size=target_size)
    imgs = image.img_to_array(img)
    imgs = np.expand_dims(imgs, axis=0)
    return imgs


def write_summary_to_file(model: keras.Model, filename: str):
    with open(filename, "w") as f:
        model.summary(print_fn=lambda x: f.write(f"{x}\n"))


def plot_histogram(prefix: str, arr: np.ndarray):
    title = textwrap.fill(prefix.replace("&", " "), 70)
    fig = plot.neuron_histogram(arr, title)
    fig.savefig(f"{prefix}-histogram.png", dpi=200)


def plot_featuremap(prefix: str, arr: np.ndarray):
    title = textwrap.fill(prefix.replace("&", " "), 70)
    fig = plot.featuremap(arr[0], title)
    fig.savefig(f"{prefix}-featuremap.png", dpi=200)


def write_tensor_video(
    model_config: ModelConfig,
    model: keras.Model,
    tensor_layout: TensorLayout,
    tiled_layout: TiledArrayLayout,
):
    prefix = prefix_of(model_config)
    preprocess_input = get_preprocessor(model_config.model)
    test_images = single_input_image("imgvideo.jpg", target_size=None)
    test_inputs = preprocess_input(test_images)
    pred_idx = analysis_client_final(model_config)

    def frames():
        _, _, w, _ = model.input_shape
        for x in range(100):
            inputs = test_inputs[..., :, x : x + w, :]
            preds = model.predict(inputs)
            pred = preds[pred_idx][0]
            yield pred

    write_video(f"{prefix}-encoder.mp4", frames(), tensor_layout, tiled_layout)


def read_tensor_video(
    model_config: ModelConfig,
    model: keras.Model,
    tensor_layout: TensorLayout,
    tiled_layout: TiledArrayLayout,
):
    prefix = prefix_of(model_config)
    prediction_decoder = imagenet_utils.decode_predictions
    frames = read_video(f"{prefix}-encoder.mp4", tensor_layout, tiled_layout)

    for i, tensor in enumerate(frames):
        tensor = np.expand_dims(tensor, axis=0)
        pred = model.predict(tensor)
        print(i, prediction_decoder(pred))


def analysis_client_final(model_config: ModelConfig) -> int:
    """Index of final client output tensor."""
    return 0 if model_config.encoder == "None" else 1


def run_analysis(
    model_config: ModelConfig,
    model_analysis: keras.Model,
    model_server: keras.Model,
    test_inputs,
    targets,
):
    prefix = prefix_of(model_config)

    pred_analysis = model_analysis.predict(test_inputs)
    pred_split = pred_analysis[0]
    pred_server = pred_analysis[-1]

    prediction_decoder = imagenet_utils.decode_predictions
    print(f"Prediction loss: {cross_entropy(pred_server, targets)}")
    print("Decoded predictions:")
    pprint(prediction_decoder(pred_server))
    print("Decoded targets:")
    pprint(prediction_decoder(targets))

    if len(pred_analysis) == 2:
        np.save(f"{prefix}-split.npy", pred_split)
        np.save(f"{prefix}-server.npy", pred_server)
        plot_histogram(f"{prefix}-split", pred_split)
        plot_featuremap(f"{prefix}-split", pred_split)
        return

    if len(pred_analysis) != 4:
        raise ValueError("Incorrect number of model outputs")

    pred_encoder = pred_analysis[1]
    pred_decoder = pred_analysis[2]

    np.save(f"{prefix}-encoder.npy", pred_encoder)
    np.save(f"{prefix}-decoder.npy", pred_decoder)
    plot_histogram(f"{prefix}-encoder", pred_encoder)
    plot_histogram(f"{prefix}-decoder", pred_decoder)
    plot_featuremap(f"{prefix}-encoder", pred_encoder)
    plot_featuremap(f"{prefix}-decoder", pred_decoder)

    client_final_idx = analysis_client_final(model_config)
    _, h, w, c = model_analysis.output_shape[client_final_idx]
    dtype = model_server.layers[0].dtype
    tensor_layout = TensorLayout(dtype, c, h, w, "hwc")
    tiled_layout = determine_tile_layout(tensor_layout)
    layouts = tensor_layout, tiled_layout
    return

    write_tensor_video(model_config, model_analysis, *layouts)
    read_tensor_video(model_config, model_server, *layouts)

    # TODO extract to "analysis.py" or similar
    # TODO analysis/experiments:
    # static component + dynamic component... try to separate?
    # take mean + std of various random images in data set, and see how feature map responds (neuron by neuron)
    # see if turning on/off neurons has effect
    # see how rotation affects things
    # reconstruction error from encoder/decoder
    # sensitivity analysis (perturb input, see how client tensor changes)
    # top-k accuracy on data set


def delete_file(filename: str):
    with suppress(FileNotFoundError):
        os.remove(filename)


def run_split(
    model: keras.Model,
    model_name: str,
    model_config: ModelConfig,
    test_inputs,
    targets,
    clean: bool = False,
    graph_plot: bool = False,
):
    print(f"run_split({model_config})")
    assert model_name == model_config.model
    prefix = prefix_of(model_config)

    if model_config.layer == "server":
        return

    if clean:
        delete_file(f"{prefix}-client.h5")
        delete_file(f"{prefix}-client.npy")
        delete_file(f"{prefix}-client.png")
        delete_file(f"{prefix}-client.tflite")
        delete_file(f"{prefix}-server.h5")
        delete_file(f"{prefix}-server.png")

    if model_config.layer == "client":
        model_client = copy_model(model)
        if not os.path.exists(f"{prefix}-client.tflite"):
            convert_to_tflite_model(model_client, f"{prefix}-client.tflite")
        del model_client
        gc.collect()
        return

    # Load and save split model
    model_client, model_server, model_analysis = split_model_by_config(
        model, model_config
    )

    if not os.path.exists(f"{prefix}-client.h5"):
        model_client.save(f"{prefix}-client.h5")
    if not os.path.exists(f"{prefix}-server.h5"):
        model_server.save(f"{prefix}-server.h5")

    if graph_plot:
        plot_model(model_client, to_file=f"{prefix}-client.png")
        plot_model(model_server, to_file=f"{prefix}-server.png")
    write_summary_to_file(model_client, f"{prefix}-client.txt")
    write_summary_to_file(model_server, f"{prefix}-server.txt")
    run_analysis(
        model_config, model_analysis, model_server, test_inputs, targets
    )
    if not os.path.exists(f"{prefix}-client.tflite"):
        convert_to_tflite_model(model_client, f"{prefix}-client.tflite")
    del model_client
    del model_server
    del model_analysis
    gc.collect()

    print("")


def run_splits(
    model_name: str,
    model_configs: Iterator[ModelConfig],
    clean_model: bool = False,
    clean_splits: bool = False,
    graph_plot: bool = False,
):
    print(f"run_splits({model_name})\n")
    prefix = prefix_of_name(model_name)

    if clean_model:
        delete_file(f"{prefix}-full.h5")
        delete_file(f"{prefix}-full.txt")

    preprocess_input = get_preprocessor(model_name)
    test_images = single_input_image("sample.jpg")
    test_inputs = preprocess_input(test_images)

    # Load and save entire model
    try:
        model = keras.models.load_model(f"{prefix}-full.h5")
    except OSError:
        os.makedirs(f"models/{model_name}", exist_ok=True)
        model = create_model(model_name)
        model.save(f"{prefix}-full.h5")
        # Force usage of tf.keras.Model which has Nodes linked correctly
        model = keras.models.load_model(f"{prefix}-full.h5")

    if graph_plot:
        plot_model(model, to_file=f"{prefix}-full.png")
    write_summary_to_file(model, f"{prefix}-full.txt")
    targets = model.predict(test_inputs)
    del model
    gc.collect()

    for model_config in model_configs:
        if model_config.layer == "server":
            continue

        # TODO If this can be avoided, it would speed things up considerably...
        # Force usage of tf.keras.Model which has Nodes linked correctly
        model = keras.models.load_model(f"{prefix}-full.h5")

        run_split(
            model,
            model_name,
            model_config,
            test_inputs,
            targets,
            clean=clean_splits,
            graph_plot=graph_plot,
        )

        print("----------\n")

        del model
        gc.collect()
        K.clear_session()

    print("\n==========\n")


def main():
    with open("models.json") as f:
        models = json.load(f)

    models = {
        model: [ModelConfig(model=model, **x) for x in opt_list]
        for model, opt_list in models.items()
    }

    os.makedirs("models", exist_ok=True)

    # Single test
    # model_name = "resnet34"
    # run_splits(model_name, models[model_name])
    # return

    for model_name, model_configs in models.items():
        run_splits(
            model_name, model_configs, clean_splits=True, graph_plot=False
        )


if __name__ == "__main__":
    main()
