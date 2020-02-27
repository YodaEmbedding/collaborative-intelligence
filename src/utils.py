from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from src.lib.layers import decoders, encoders
from src.lib.predecode import *
from src.lib.split import split_model
from src.lib.tile import determine_tile_layout
from src.modelconfig import ModelConfig, PostencoderConfig


def get_predecoder(
    postencoder_config: PostencoderConfig,
    model_config: ModelConfig,
    tensor_layout: TensorLayout,
) -> Predecoder:
    encoder_type = model_config.encoder
    postencoder_type = postencoder_config.type

    if model_config.layer == "client":
        assert encoder_type == "None"
        assert postencoder_type == "None"
        shape = (-1,)
        dtype = np.float32
        return TensorPredecoder(shape, dtype)

    shape = tensor_layout.shape
    dtype = tensor_layout.dtype

    if model_config.layer == "server":
        assert encoder_type == "None"
        if postencoder_type == "None":
            assert shape[-1] == 3
            return RgbPredecoder(shape, dtype)
        if postencoder_type == "jpeg":
            return JpegRgbPredecoder(tensor_layout)
        raise ValueError("Unknown postencoder")

    if encoder_type == "None":
        assert postencoder_type == "None"
        return TensorPredecoder(shape, dtype)

    if encoder_type == "UniformQuantizationU8Encoder":
        assert dtype == np.uint8
        if postencoder_type == "None":
            return TensorPredecoder(shape, dtype)
        if postencoder_type == "jpeg":
            tiled_layout = determine_tile_layout(tensor_layout)
            return JpegPredecoder(tiled_layout, tensor_layout)
        raise ValueError("Unknown postencoder")

    raise ValueError("Unknown encoder")


def split_model_by_config(
    model: keras.Model, model_config: ModelConfig,
) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """Split model by given layer name.

    Attaches encoder layer to end of client model. Attaches decoder
    layer to beginning of server model.

    Returns:
        model_client
        model_server
        model_analysis
    """
    layer = model_config.layer
    encoder = (
        None
        if model_config.encoder == "None"
        else encoders[model_config.encoder](**model_config.encoder_args)
    )
    decoder = (
        None
        if model_config.decoder == "None"
        else decoders[model_config.decoder](**model_config.decoder_args)
    )
    return split_model(model, layer, encoder, decoder)
