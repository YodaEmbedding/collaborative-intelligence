import errno
import gc
import json
import os
import textwrap
from contextlib import suppress
from pprint import pprint
from typing import Callable, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # pylint: disable=import-error
from classification_models.tfkeras import Classifiers
from tensorflow import keras
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import plot_model

from layers import decoders, encoders
from modelconfig import ModelConfig
from split import split_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def convert_to_tflite_model(
    keras_model_filename: str, tflite_filename: str, **kwargs
):
    """Convert keras model file to tflite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model_file(
        keras_model_filename, **kwargs
    )
    tflite_model = converter.convert()
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)


def create_directory(directory: str):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def create_model(model_name: str) -> keras.Model:
    """Model factory."""
    shape = (224, 224, 3)
    model_creator, _ = Classifiers.get(model_name)
    return model_creator(shape, weights="imagenet")


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    return -(np.sum(targets * np.log(predictions))) / predictions.shape[0]


def get_preprocessor(model_name: str):
    """Get input preprocessor for model."""
    return Classifiers.get(model_name)[1]


def single_input_image(filename: str):
    """Load single image for testing."""
    img = image.load_img(filename, target_size=(224, 224))
    imgs = image.img_to_array(img)
    imgs = np.expand_dims(imgs, axis=0)
    return imgs


def write_summary_to_file(model: keras.Model, filename: str):
    with open(filename, "w") as f:
        model.summary(print_fn=lambda x: f.write(f"{x}\n"))


def save_histogram(prefix: str, arr: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print(f"Min neuron: {np.min(arr)}")
    print(f"Max neuron: {np.max(arr)}")
    ax.hist(arr, bins=np.linspace(np.min(arr), np.max(arr), 20))
    ax.set_xlabel("Neuron value")
    ax.set_ylabel("Frequency")
    ax.set_title(textwrap.fill(prefix, 70), fontsize="xx-small")
    fig.savefig(f"{prefix}-histogram.png", dpi=200)


def run_analysis(
    prefix: str,
    model_client: keras.Model,
    model_server: keras.Model,
    test_inputs,
    targets,
):
    pred_client = model_client.predict(test_inputs)
    pred_server = model_server.predict(pred_client)

    np.save(f"{prefix}.npy", pred_client)
    save_histogram(prefix, pred_client.reshape((-1,)))

    print(f"Prediction loss: {cross_entropy(pred_server, targets)}")
    prediction_decoder = imagenet_utils.decode_predictions
    print("Decoded predictions:")
    pprint(prediction_decoder(pred_server))
    print("Decoded targets:")
    pprint(prediction_decoder(targets))

    # TODO reconstruction error from encoder/decoder
    # TODO sensitivity analysis (perturb input, see how client tensor changes)
    # TODO top-k accuracy on data set


def run_split(
    model: keras.Model,
    model_name: str,
    model_config: ModelConfig,
    test_inputs,
    targets,
    clean: bool = False,
):
    print(f"run_split({model_config})")
    prefix = model_config.to_path()

    if clean:
        with suppress(FileNotFoundError):
            os.remove(f"{prefix}-client.h5")
            os.remove(f"{prefix}-client.npy")
            os.remove(f"{prefix}-client.png")
            os.remove(f"{prefix}-client.tflite")
            os.remove(f"{prefix}-server.h5")
            os.remove(f"{prefix}-server.png")

    client_objects = {}
    server_objects = {}
    if model_config.encoder != "None":
        client_objects[model_config.encoder] = encoders[model_config.encoder]
    if model_config.decoder != "None":
        server_objects[model_config.decoder] = decoders[model_config.decoder]

    # Load and save split model
    model_client, model_server, model_analysis = split_model(
        model, model_config
    )

    if not os.path.exists(f"{prefix}-client.h5"):
        model_client.save(f"{prefix}-client.h5")
    if not os.path.exists(f"{prefix}-server.h5"):
        model_server.save(f"{prefix}-server.h5")

    plot_model(model_client, to_file=f"{prefix}-client.png")
    plot_model(model_server, to_file=f"{prefix}-server.png")
    write_summary_to_file(model, f"{prefix}-client.txt")
    write_summary_to_file(model, f"{prefix}-server.txt")
    run_analysis(
        f"{prefix}-client", model_client, model_server, test_inputs, targets
    )
    if not os.path.exists(f"{prefix}-client.tflite"):
        convert_to_tflite_model(
            f"{prefix}-client.h5",
            f"{prefix}-client.tflite",
            custom_objects=client_objects,
        )
    del model_client
    del model_server
    gc.collect()
    K.clear_session()

    print("")


def run_splits(
    model_name: str,
    model_configs: Iterable[ModelConfig],
    clean_model: bool = False,
    clean_splits: bool = False,
):
    print(f"run_splits({model_name})\n")
    prefix = f"{model_name}/{model_name}"

    if clean_model:
        with suppress(FileNotFoundError):
            os.remove(f"{prefix}-full.h5")
            os.remove(f"{prefix}-full.tflite")
            os.remove(f"{prefix}-full.txt")

    preprocess_input = get_preprocessor(model_name)
    test_images = single_input_image("sample.jpg")
    test_inputs = preprocess_input(test_images)

    # Load and save entire model
    try:
        model = keras.models.load_model(f"{prefix}-full.h5")
    except OSError:
        create_directory(model_name)
        model = create_model(model_name)
        model.save(f"{prefix}-full.h5")
        # Force usage of tf.keras.Model which has Nodes linked correctly
        model = keras.models.load_model(f"{prefix}-full.h5")

    plot_model(model, to_file=f"{prefix}-full.png")
    write_summary_to_file(model, f"{prefix}-full.txt")
    targets = model.predict(test_inputs)
    if not os.path.exists(f"{prefix}-client.tflite"):
        convert_to_tflite_model(f"{prefix}-full.h5", f"{prefix}-full.tflite")
    del model
    gc.collect()

    for model_config in model_configs:
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

    # Single test
    # model_name = "vgg16"
    # run_splits(model_name, models[model_name])
    # return

    for model_name, model_configs in models.items():
        run_splits(model_name, model_configs, clean_splits=True)


if __name__ == "__main__":
    main()


# TODO plot analysis (e.g. histogram, tensor data file save, n-bit compression accuracies, etc)
# TODO delete resnet-keras-split
