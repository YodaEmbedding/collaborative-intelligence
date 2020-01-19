import gc
from typing import List

from tensorflow import keras

from src.modelconfig import ModelConfig


def prefix_of(model_config: ModelConfig) -> str:
    return f"models/{model_config.to_path()}"


def release_models(*models: List[keras.Model]):
    for model in models:
        del model
    gc.collect()
    keras.backend.clear_session()
    gc.collect()
