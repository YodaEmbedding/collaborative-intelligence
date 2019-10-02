import errno
import gc
import json
import os
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

from split import (
    SplitOptions,
    UniformQuantizationU8Decoder,
    UniformQuantizationU8Encoder,
    split_model,
)

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
    ax.hist(arr, bins=np.linspace(np.min(arr), np.max(arr), 20))
    ax.set_xlabel("Neuron value")
    ax.set_ylabel("Frequency")
    fig.savefig(f"{prefix}-histogram.png")


def run_analysis(prefix: str, arr: np.ndarray):
    np.save(f"{prefix}.npy", arr)
    save_histogram(prefix, arr.reshape((-1,)))


def run_split(
    model: keras.Model,
    model_name: str,
    split_options: SplitOptions,
    test_inputs,
    targets,
    clean: bool = False,
):
    print(f"run_split({model_name}, {split_options})")
    prefix = f"{model_name}/{model_name}-{split_options}"

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
    if split_options.encoder is not None:
        enc = split_options.encoder.__class__
        client_objects = {enc.__name__: enc}
    if split_options.decoder is not None:
        dec = split_options.decoder.__class__
        server_objects = {dec.__name__: dec}

    # Load and save split model
    try:
        model_client = keras.models.load_model(
            f"{prefix}-client.h5", custom_objects=client_objects
        )
        model_server = keras.models.load_model(
            f"{prefix}-server.h5", custom_objects=server_objects
        )
    except OSError:
        model_client, model_server = split_model(model, split_options)
        model_client.save(f"{prefix}-client.h5")
        model_server.save(f"{prefix}-server.h5")

    plot_model(model_client, to_file=f"{prefix}-client.png")
    plot_model(model_server, to_file=f"{prefix}-server.png")
    write_summary_to_file(model, f"{prefix}-client.txt")
    write_summary_to_file(model, f"{prefix}-server.txt")
    predictions = model_client.predict(test_inputs)
    run_analysis(f"{prefix}-client", predictions)
    predictions = model_server.predict(predictions)
    if not os.path.exists(f"{prefix}-client.tflite"):
        convert_to_tflite_model(
            f"{prefix}-client.h5",
            f"{prefix}-client.tflite",
            custom_objects=client_objects,
        )
    del model_client
    del model_server
    gc.collect()

    print(f"Prediction loss: {cross_entropy(predictions, targets)}")
    pred_decoder = imagenet_utils.decode_predictions
    decoded_predictions = pred_decoder(predictions)
    decoded_targets = pred_decoder(targets)
    print("Decoded predictions:")
    pprint(decoded_predictions)
    print("Decoded targets:")
    pprint(decoded_targets)

    print("")


def run_splits(
    model_name: str,
    split_options_list: Iterable[SplitOptions],
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

    for split_options in split_options_list:
        # Force usage of tf.keras.Model which has Nodes linked correctly
        model = keras.models.load_model(f"{prefix}-full.h5")

        run_split(
            model,
            model_name,
            split_options,
            test_inputs,
            targets,
            clean=clean_splits,
        )

        del model
        gc.collect()
        K.clear_session()

    print("\n----------\n")


def main():
    with open("models.json") as f:
        models = json.load(f)

    refs = [UniformQuantizationU8Encoder, UniformQuantizationU8Decoder]
    refs = {x.__name__: x for x in refs}
    refs["None"] = lambda: None
    print(refs)

    models = {
        model: [
            SplitOptions(
                x["layer"],
                refs[x["encoder"]](**x.get("encoder_args", {})),
                refs[x["decoder"]](**x.get("decoder_args", {})),
            )
            for x in opt_list
        ]
        for model, opt_list in models.items()
    }

    # Single test
    # model_name = "vgg16"
    # run_splits(model_name, models[model_name])
    # return

    for model_name, split_options_list in models.items():
        run_splits(model_name, split_options_list, clean_splits=True)


if __name__ == "__main__":
    main()


# TODO plot analysis (e.g. histogram, tensor data file save, n-bit compression accuracies, etc)
# TODO delete resnet-keras-split
