from dataclasses import dataclass
from typing import (
    Awaitable,
    ByteString,
    Dict,
    Generator,
    Generic,
    List,
    Tuple,
    TypeVar,
)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import imagenet_utils

from src.layers import decoders
from src.modelconfig import ModelConfig


@dataclass
class ModelReference:
    ref_count: int
    model: keras.Model


class ModelManager:
    """Manages TensorFlow models.

    Holds model references, loads, releases, and runs predictions.
    """

    def __init__(self):
        self.models: Dict[ModelConfig, ModelReference] = {}

    def acquire(self, model_config: ModelConfig):
        if model_config not in self.models:
            print(f"Loading model {model_config}")
            model = self._load_model(model_config)
            self.models[model_config] = ModelReference(0, model)
            print(f"Loaded model {model_config}")
        self.models[model_config].ref_count += 1

    def release(self, model_config: ModelConfig):
        self.models[model_config].ref_count -= 1
        if self.models[model_config].ref_count == 0:
            print(f"Releasing model {model_config}")
            # del self.models[model_config].model
            # del self.models[model_config]
            print(f"Released model {model_config}")

    def decode_data(
        self, model_config: ModelConfig, data: ByteString
    ) -> np.ndarray:
        model = self.models[model_config].model
        data_tensor = _decode_data(model, data)
        return data_tensor

    # @synchronized
    def predict(
        self, model_config: ModelConfig, data_tensor: np.ndarray
    ) -> List[Tuple[str, str, float]]:
        model = self.models[model_config].model
        predictions = (
            model.predict(data_tensor) if model is not None else data_tensor
        )
        return self._decode_predictions(predictions)

    def _decode_predictions(
        self, predictions: np.ndarray, num_preds: int = 3
    ) -> List[Tuple[str, str, float]]:
        pred = imagenet_utils.decode_predictions(predictions)[0][:num_preds]
        return [(name, desc, float(score)) for name, desc, score in pred]

    def _load_model(self, model_config: ModelConfig) -> keras.Model:
        model_name = model_config.model
        if model_config.layer == "server":
            return keras.models.load_model(
                filepath=f"models/{model_name}/{model_name}-full.h5"
            )
        if model_config.layer == "client":
            return None
        decoder = model_config.decoder
        custom_objects = {}
        if decoder != "None":
            custom_objects[decoder] = decoders[decoder]
        return keras.models.load_model(
            filepath=f"models/{model_config.to_path()}-server.h5",
            custom_objects=custom_objects,
        )


def _decode_data(model: keras.Model, data: ByteString) -> np.ndarray:
    # Handle client side
    if model is None:
        # TODO np.float32? should probably store more info about model
        return np.frombuffer(data, dtype=np.float32)[None, ...]
    input_shape = model.layers[1].input_shape[1:]
    input_type = model.layers[0].dtype  # TODO verify
    dtype = _to_np_dtype(input_type)
    return np.frombuffer(data, dtype=dtype).reshape((-1, *input_shape))


def _to_np_dtype(dtype):
    return {
        "float32": np.float32,
        "uint8": np.uint8,
        tf.float32: np.float32,
        tf.uint8: np.uint8,
    }[dtype]
