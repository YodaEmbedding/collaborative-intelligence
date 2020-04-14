import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras

from src.analysis import dataset
from src.analysis.utils import (
    basename_of,
    compile_kwargs,
    get_cut_layers,
    title_of,
)
from src.lib.layouts import TensorLayout
from src.lib.split import split_model


class ExperimentRunner:
    def __init__(self, model_name, layer_name, dataset_size, batch_size):
        self.model_name = model_name
        self.layer_name = layer_name
        self.dataset_size = dataset_size
        self.batch_size = batch_size

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

        self.data = dataset.dataset().take(dataset_size)
        self.data_batched = self.data.batch(batch_size)
        labels = self.data.map(lambda x, y: y)
        self.data_labels = np.array(list(tfds.as_numpy(labels)))
        self.data_numpy_batches = list(tfds.as_numpy(self.data_batched))

        self.d = {}

    def client_tensor_batches(self, images: bool = False, copy: bool = True):
        client_tensors = self.d["tensors"]
        bs = self.batch_size
        for i, (frames, labels) in enumerate(self.data_numpy_batches):
            client_tensors_batch = client_tensors[i * bs : (i + 1) * bs]
            if copy:
                client_tensors_batch = client_tensors_batch.copy()
            if images:
                yield client_tensors_batch, frames, labels
                continue
            yield client_tensors_batch, labels

    def summarize(self):
        for k, v in self.d.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (float, np.float32, np.float64)):
                print(f"{k}: {v:.4g}")
                continue
            print(f"{k}: {v}")
