import gc
import json
import textwrap
import urllib.request
from io import BytesIO
from typing import ByteString, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from src.layouts import TensorLayout
from src.modelconfig import ModelConfig, PostencoderConfig
from src.predecode import get_predecoder
from src.split import split_model
from src.tile import determine_tile_layout, tile

BATCH_SIZE = 16
BYTES_PER_KB = 1000
data_dir = "data"
csv_path = f"{data_dir}/data.csv"
compile_kwargs = {
    "loss": "sparse_categorical_crossentropy",
    "optimizer": keras.optimizers.RMSprop(),
    "metrics": ["accuracy"],
}


def release_models(*models: List[keras.Model]):
    for model in models:
        del model
    gc.collect()
    keras.backend.clear_session()
    gc.collect()


def prefix_of(model_config: ModelConfig) -> str:
    return f"models/{model_config.to_path()}"


def get_imagenet_labels() -> Dict[str, List[str]]:
    try:
        with open("imagenet_class_index.json", "r") as f:
            json_data = f.read()
    except FileNotFoundError:
        url = (
            "https://storage.googleapis.com/download.tensorflow.org/data/"
            "imagenet_class_index.json"
        )
        response = urllib.request.urlopen(url)
        json_data = response.read()
        with open("imagenet_class_index.json", "wb") as f:
            f.write(json_data)

    return json.loads(json_data)


def get_imagenet_reverse_lookup() -> Dict[str, int]:
    d = get_imagenet_labels()
    return {name: int(idx) for idx, (name, label) in d.items()}


imagenet_lookup = get_imagenet_reverse_lookup()


def dataset_from(df: pd.DataFrame, kb: int):
    filepaths = df["file"].map(lambda x: f"{data_dir}/{kb}kb/{x}")
    labels = df["label"].replace(imagenet_lookup)
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(parse_row).batch(BATCH_SIZE)
    return dataset


def parse_row(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw)
    return img, label


def jpeg_encode(arr: np.ndarray, quality: int) -> ByteString:
    img = Image.fromarray(arr)
    with BytesIO() as buf:
        img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return buf.read()


def pad(img: np.ndarray) -> np.ndarray:
    MBU_SIZE = 16
    py = -img.shape[0] % MBU_SIZE
    px = -img.shape[1] % MBU_SIZE
    return np.pad(img, ((0, py), (0, px), (0, 0)))


def postencode(
    client_tensor: np.ndarray, tensor_layout: TensorLayout, quality: int
) -> ByteString:
    tiled_layout = determine_tile_layout(tensor_layout)
    tiled_tensor = tile(client_tensor, tensor_layout, tiled_layout)
    tiled_tensor = np.tile(tiled_tensor[..., np.newaxis], 3)
    tiled_tensor = pad(tiled_tensor)
    client_bytes = jpeg_encode(tiled_tensor, quality)
    return client_bytes


def predecode(
    client_bytes: ByteString,
    model_config: ModelConfig,
    tensor_layout: TensorLayout,
) -> np.ndarray:
    postencoder_config = PostencoderConfig("jpeg", 100)
    predecoder = get_predecoder(
        postencoder_config, model_config, tensor_layout
    )
    return predecoder.run(client_bytes)


def make_quality_per_kb_table(client_tensors):
    quality_lookup = []
    for client_tensor in client_tensors:
        tensor_layout = TensorLayout.from_tensor(client_tensor, "hwc")
        d = {}
        for quality in range(1, 101):
            encoded_bytes = postencode(client_tensor, tensor_layout, quality)
            kb = int(len(encoded_bytes) / BYTES_PER_KB)
            if kb not in d:
                d[kb] = quality
        quality_lookup.append(d)
    return quality_lookup


def evaluate_accuracy(df: pd.DataFrame, model: keras.Model, kb: int) -> float:
    dataset = dataset_from(df, kb)
    loss, accuracy = model.evaluate(dataset, verbose=0)
    return accuracy


def evaluate_accuracy_server(
    df: pd.DataFrame,
    model_config: ModelConfig,
    model_server: keras.Model,
    kb: int,
    client_tensors: np.ndarray,
    quality_lookup: List[Dict[int, int]],
) -> float:
    labels = df["label"].replace(imagenet_lookup)
    samples = []
    for i, (client_tensor, label) in enumerate(zip(client_tensors, labels)):
        tensor_layout = TensorLayout.from_tensor(client_tensor, "hwc")
        quality = quality_lookup[i].get(kb, None)
        if quality is None:
            continue
        encoded_bytes = postencode(client_tensor, tensor_layout, quality)
        decoded_tensor = predecode(encoded_bytes, model_config, tensor_layout)
        samples.append((decoded_tensor, label))
    if len(samples) == 0:
        return None
    decoded_tensors = [t for t, l in samples]
    decoded_tensors = np.array(decoded_tensors)
    labels = [l for t, l in samples]
    dataset = tf.data.Dataset.from_tensor_slices((decoded_tensors, labels))
    dataset = dataset.batch(BATCH_SIZE)
    loss, accuracy = model_server.evaluate(dataset, verbose=0)
    return accuracy


def evaluate_accuracy_shared(
    df: pd.DataFrame,
    model_config: ModelConfig,
    model_client: keras.Model,
    model_server: keras.Model,
) -> List[Tuple[int, float]]:
    dataset = dataset_from(df, 30)
    client_tensors = model_client.predict(dataset)
    quality_lookup = make_quality_per_kb_table(client_tensors)
    accuracy_shared = []

    for kb in range(1, 31):
        accuracy = evaluate_accuracy_server(
            df, model_config, model_server, kb, client_tensors, quality_lookup
        )
        if accuracy is None:
            continue
        accuracy_shared.append((kb, accuracy))

    return accuracy_shared


def plot_accuracy_vs_kb(
    prefix: str,
    accuracy_server: List[Tuple[int, float]],
    accuracy_shared: List[Tuple[int, float]],
):
    kbs_server = np.array([k for (k, _) in accuracy_server]) / 1.024
    acc_server = np.array([a for (_, a) in accuracy_server])
    kbs_shared = np.array([k for (k, _) in accuracy_shared]) / 1.024
    acc_shared = np.array([a for (_, a) in accuracy_shared])

    title = textwrap.fill(prefix.replace("&", " "), 70)
    plt.figure()
    plt.plot(kbs_server, acc_server, label="server-only inference")
    plt.plot(kbs_shared, acc_shared, label="shared inference")
    ax: plt.Axes = plt.gca()
    ax.legend()
    ax.set_xlim([0, 10])
    ax.set_xlabel("KB/frame")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize="xx-small")
    plt.savefig(f"{prefix}-accuracy_vs_kb.png", dpi=200)


def analyze_accuracy_vs_kb(
    df: pd.DataFrame, model: keras.Model, model_configs: List[ModelConfig]
):
    accuracy_server = [
        (kb, evaluate_accuracy(df, model, kb)) for kb in range(1, 31)
    ]

    encoders = ["UniformQuantizationU8Encoder"]
    model_configs = [x for x in model_configs if x.encoder in encoders]

    for model_config in model_configs:
        prefix = prefix_of(model_config)
        accuracy_shared = analyze_model_config(df, model, model_config)
        plot_accuracy_vs_kb(prefix, accuracy_server, accuracy_shared)


def analyze_model_config(
    df: pd.DataFrame, model: keras.Model, model_config: ModelConfig
) -> List[Tuple[int, float]]:
    model_client, model_server, model_analysis = split_model(
        model, model_config
    )
    model_server.compile(**compile_kwargs)
    accuracy_shared = evaluate_accuracy_shared(
        df, model_config, model_client, model_server
    )
    release_models(model_client, model_server, model_analysis)
    return accuracy_shared


def analyze_model(
    df: pd.DataFrame, model_name: str, model_configs: List[ModelConfig]
):
    model_path = f"models/{model_name}/{model_name}-full.h5"
    model: keras.Model = keras.models.load_model(model_path)
    model.compile(**compile_kwargs)
    analyze_accuracy_vs_kb(df, model, model_configs)
    release_models(model)


def main():
    df = pd.read_csv(csv_path)

    with open("models.json") as f:
        d = json.load(f)

    model_configs = {
        model_name: [
            ModelConfig(model=model_name, **config_dict)
            for config_dict in config_dicts
        ]
        for model_name, config_dicts in d.items()
    }

    for model_name, model_configs_ in model_configs.items():
        analyze_model(df, model_name, model_configs_)


if __name__ == "__main__":
    main()


# TODO move generate_models analysis stuff here...
