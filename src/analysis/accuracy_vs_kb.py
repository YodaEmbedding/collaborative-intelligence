import textwrap
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.analysis.dataset import dataset_kb
from src.analysis.utils import prefix_of, release_models
from src.layouts import TensorLayout
from src.modelconfig import ModelConfig, PostencoderConfig
from src.postencode import JpegPostencoder, Postencoder
from src.predecode import JpegPredecoder, Predecoder
from src.split import split_model

BATCH_SIZE = 64
BYTES_PER_KB = 1000
data_dir = "data"
csv_path = f"{data_dir}/data.csv"


def compute_dataset_accuracies(
    model_client: keras.Model,
    model_server: keras.Model,
    postencoder: Postencoder,
    predecoder: Predecoder,
    dataset: tf.data.Dataset,
    accuracy_func: Callable[[np.ndarray, int], float],
) -> List[float]:
    accuracies = []

    for frames, labels in dataset:
        client_tensors = model_client.predict(frames)
        decoded_tensors = []
        for client_tensor in client_tensors:
            encoded_bytes = postencoder.run(client_tensor)
            decoded_tensor = predecoder.run(encoded_bytes)
            decoded_tensors.append(decoded_tensor)
        decoded_tensors = np.array(decoded_tensors)
        predictions = model_server.predict(decoded_tensors)
        accuracies.extend(accuracy_func(labels.numpy(), predictions))

    return accuracies


def analyze_accuracy_vs_kb(
    model: keras.Model, model_configs: List[ModelConfig]
):
    accuracies_server = [
        (kb, _evaluate_accuracy_kb(model, kb)) for kb in range(1, 31)
    ]
    encoders = ["UniformQuantizationU8Encoder"]
    model_configs = [x for x in model_configs if x.encoder in encoders]

    for model_config in model_configs:
        print(f"Analyzing accuracy vs KB\n{model_config}\n")
        prefix = prefix_of(model_config)
        model_client, model_server, model_analysis = split_model(
            model, model_config
        )
        accuracies_shared = _evaluate_accuracies_shared_kb(
            model_client, model_server
        )
        _plot_accuracy_vs_kb(prefix, accuracies_server, accuracies_shared)
        release_models(model_client, model_server, model_analysis)


def _evaluate_accuracy_kb(model: keras.Model, kb: int,) -> float:
    dataset = dataset_kb(kb)
    predictions = model.predict(dataset.batch(BATCH_SIZE))
    labels = np.array(list(dataset.map(lambda x, l: l)))
    accuracies = _categorical_top1_accuracy(labels, predictions)
    return np.mean(accuracies)


def _evaluate_accuracies_shared_kb(
    model_client: keras.Model, model_server: keras.Model,
) -> List[Tuple[int, float]]:
    accuracies_shared = defaultdict(list)
    dataset = dataset_kb().batch(BATCH_SIZE)
    client_tensors = model_client.predict(dataset)
    quality_lookup = _make_quality_lut(client_tensors)
    kb_lookup = [{q: kb for kb, q in d.items()} for d in quality_lookup]
    tensor_layout = TensorLayout.from_shape(
        client_tensors.shape[1:], "hwc", client_tensors.dtype
    )

    for quality in range(1, 100):
        keep = [quality in d for d in kb_lookup]
        keep_ds = tf.data.Dataset.from_tensor_slices(keep)

        dataset_quality = (
            tf.data.Dataset.zip((dataset, keep_ds))
            .filter(lambda x, c: c)
            .map(lambda x, c: x)
        )

        postencoder_config = PostencoderConfig("jpeg", quality)
        postencoder = JpegPostencoder(tensor_layout, postencoder_config)
        tiled_layout = postencoder.tiled_layout
        predecoder = JpegPredecoder(tiled_layout, tensor_layout)

        accuracies = compute_dataset_accuracies(
            model_client,
            model_server,
            postencoder,
            predecoder,
            dataset_quality,
            _categorical_top1_accuracy,
        )

        kbs = [d[quality] for d in kb_lookup if quality in d]
        for kb, acc in zip(kbs, accuracies):
            accuracies_shared[kb].append(acc)

    return [
        (kb, np.mean(accuracies_shared[kb]))
        for kb in range(1, 31)
        if len(accuracies_shared[kb]) != 0
    ]


def _make_quality_lut(
    client_tensors: List[np.ndarray],
) -> List[Dict[int, int]]:
    quality_lookup = []
    for client_tensor in client_tensors:
        tensor_layout = TensorLayout.from_tensor(client_tensor, "hwc")
        d = {}
        for quality in range(1, 101):
            postencoder_config = PostencoderConfig("jpeg", quality)
            postencoder = JpegPostencoder(tensor_layout, postencoder_config)
            encoded_bytes = postencoder.run(client_tensor)
            kb = int(len(encoded_bytes) / BYTES_PER_KB)
            if kb not in d:
                d[kb] = quality
        quality_lookup.append(d)
    return quality_lookup


def _plot_accuracy_vs_kb(
    prefix: str,
    accuracies_server: List[Tuple[int, float]],
    accuracies_shared: List[Tuple[int, float]],
):
    kbs_server = np.array([k for (k, _) in accuracies_server]) / 1.024
    acc_server = np.array([a for (_, a) in accuracies_server])
    kbs_shared = np.array([k for (k, _) in accuracies_shared]) / 1.024
    acc_shared = np.array([a for (_, a) in accuracies_shared])

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


def _categorical_top1_accuracy(
    label: np.ndarray, pred: np.ndarray
) -> np.ndarray:
    return (np.argmax(pred, axis=-1) == label).astype(np.float32)
