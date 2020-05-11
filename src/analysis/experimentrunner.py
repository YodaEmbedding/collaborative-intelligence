from typing import Iterator, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from src.analysis import dataset
from src.analysis.utils import (
    basename_of,
    compile_kwargs,
    get_cut_layers,
    release_models,
    title_of,
)
from src.lib.layouts import TensorLayout
from src.lib.split import split_model


class ExperimentRunner:
    def __init__(
        self,
        model_name: str,
        layer_name: str,
        *,
        dataset_size: int,
        batch_size: int,
        test_dataset_size: int = None,
    ):
        self.model_name = model_name
        self.layer_name = layer_name

        self.model = keras.models.load_model(
            f"models/{model_name}/{model_name}-full.h5", compile=False
        )

        cut_layers = [x.name for x in get_cut_layers(self.model.layers[0])]
        i = next(i for i, x in enumerate(cut_layers) if x == layer_name)
        n = len(cut_layers)
        self.title = title_of(model_name, layer_name, i, n)
        self.basename = basename_of(model_name, layer_name, i, n)

        c, s, a = split_model(self.model, layer=self.layer_name)
        self.model_client = c
        self.model_server = s
        self.model_analysis = a
        self.model_client.compile(**compile_kwargs)
        self.model_server.compile(**compile_kwargs)

        self.tensor_layout = TensorLayout.from_shape(
            c.output_shape[1:], "hwc", c.dtype
        )

        if test_dataset_size is None:
            test_dataset_size = dataset_size

        data_all = dataset.dataset()
        data = data_all.take(dataset_size)
        data_test = data_all.skip(dataset_size).take(test_dataset_size)
        self.data = ClientTensorDataset(data, self.model_client, batch_size)
        self.data_test = ClientTensorDataset(
            data_test, self.model_client, batch_size
        )

        self.d = {}

    def close(self):
        release_models(
            self.model_client, self.model_server, self.model_analysis
        )

    def summarize(self):
        for k, v in self.d.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (float, np.float32, np.float64)):
                print(f"{k}: {v:.4g}")
                continue
            print(f"{k}: {v}")


class ClientTensorDataset:
    """Caches client tensor inferences in memory."""

    def __init__(
        self, data: tf.data.Dataset, model_client: keras.Model, batch_size: int
    ):
        self.data = data
        self.batch_size = batch_size
        self.labels = np.array(list(tfds.as_numpy(data.map(lambda x, y: y))))
        self.client_tensors = model_client.predict(data.batch(batch_size))

    def client_tensor_batches(
        self, *, images: bool = False, copy: bool = True, take: int = None
    ) -> Iterator[
        Union[
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray],
        ]
    ]:
        bs = self.batch_size
        data = self.data if take is None else self.data.take(take)
        batches = tfds.as_numpy(data.batch(bs))
        for i, (frames, labels) in enumerate(batches):
            client_tensors_batch = self.client_tensors[i * bs : (i + 1) * bs]
            if copy:
                client_tensors_batch = client_tensors_batch.copy()
            if images:
                yield client_tensors_batch, frames, labels
                continue
            yield client_tensors_batch, labels
