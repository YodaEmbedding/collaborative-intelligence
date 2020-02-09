import textwrap
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    accuracies_server = np.concatenate(
        [_evaluate_accuracy_kb(model, kb) for kb in range(1, 31)], axis=1
    )
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


def _evaluate_accuracy_kb(model: keras.Model, kb: int) -> np.ndarray:
    dataset = dataset_kb(kb)
    predictions = model.predict(dataset.batch(BATCH_SIZE))
    labels = np.array(list(dataset.map(lambda x, l: l)))
    accuracies = _categorical_top1_accuracy(labels, predictions)
    kbs = np.ones_like(accuracies) * kb
    return np.vstack((kbs, accuracies))


def _evaluate_accuracies_shared_kb(
    model_client: keras.Model, model_server: keras.Model,
) -> np.ndarray:
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

    accuracies_shared = [
        np.vstack((np.ones(len(xs)) * kb, np.array(xs)))
        for kb, xs in accuracies_shared.items()
    ]

    return np.concatenate(accuracies_shared, axis=1)


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
    accuracies_server: np.ndarray,
    accuracies_shared: np.ndarray,
):
    kbs_server = accuracies_server[0] / 1.024
    acc_server = accuracies_server[1]
    kbs_shared = accuracies_shared[0] / 1.024
    acc_shared = accuracies_shared[1]

    data_server = pd.DataFrame({"kbs": kbs_server, "acc": acc_server})
    data_shared = pd.DataFrame({"kbs": kbs_shared, "acc": acc_shared})

    data_server.to_csv(f"{prefix}-accuracy_vs_kb-server.csv")
    data_shared.to_csv(f"{prefix}-accuracy_vs_kb-shared.csv")

    title = textwrap.fill(prefix.replace("&", " "), 70)
    plt.figure()
    ax = sns.lineplot(x="kbs", y="acc", data=data_server)
    ax = sns.lineplot(x="kbs", y="acc", data=data_shared)
    ax: plt.Axes = plt.gca()
    ax.legend(
        labels=["server-only inference", "shared inference"],
        loc="lower right",
    )
    ax.set(xlim=(0, 30), ylim=(0, 1))

    # TODO move into set
    ax.set_xlabel("KB/frame")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize="xx-small")
    plt.savefig(f"{prefix}-accuracy_vs_kb.png", dpi=200)


def _categorical_top1_accuracy(
    label: np.ndarray, pred: np.ndarray
) -> np.ndarray:
    return (np.argmax(pred, axis=-1) == label).astype(np.float32)
