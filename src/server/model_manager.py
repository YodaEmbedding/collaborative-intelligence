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
from tensorflow import keras
from tensorflow.keras.applications import imagenet_utils

from src.layers import decoders
from src.modelconfig import ModelConfig
from src.predecode import to_np_dtype
from src.tile import (
    TensorLayout,
    TiledArrayLayout,
    determine_tile_layout,
    detile,
)


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
            model = _load_model(model_config)
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

    def input_tensor_layout(self, model_config: ModelConfig) -> TensorLayout:
        model = self.models[model_config].model
        # Check if client-side inference
        if model is None:
            return None
        input_dtype = to_np_dtype(model.layers[0].dtype)
        # TODO why is this layer[1]?
        input_shape = model.layers[1].input_shape[1:]
        h, w, c = input_shape
        return TensorLayout(input_dtype, c, h, w, "hwc")

    # @synchronized
    def predict(
        self, model_config: ModelConfig, data_tensor: np.ndarray
    ) -> List[Tuple[str, str, float]]:
        model = self.models[model_config].model
        if model is None:
            return data_tensor
        return model.predict(data_tensor)

    def decode_predictions(
        self,
        model_config: ModelConfig,
        predictions: np.ndarray,
        num_preds: int = 3,
    ) -> List[Tuple[str, str, float]]:
        return _decode_predictions_imagenet(predictions, num_preds)


def _decode_predictions_imagenet(
    predictions: np.ndarray, num_preds: int = 3
) -> List[Tuple[str, str, float]]:
    pred = imagenet_utils.decode_predictions(predictions)[0][:num_preds]
    return [(name, desc, float(score)) for name, desc, score in pred]


def _load_model(model_config: ModelConfig) -> keras.Model:
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
