from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.layers import Layer

from layers import decoders, encoders
from modelconfig import ModelConfig

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def split_model(
    model: keras.Model, model_config: ModelConfig
) -> Tuple[keras.Model, keras.Model]:
    """Split model by given layer index.

    Attaches encoder layer to end of client model. Attaches decoder
    layer to beginning of server model.

    Returns:
        - model_client
        - model_server
        - model_analysis
    """
    split_idx = _get_layer_idx_by_name(model, model_config.layer)
    layers = model.layers
    first_layer = layers[0]
    split_layer = layers[split_idx]

    outputs3 = []

    # Client-side (with encoder)
    inputs1 = keras.Input(_input_shape(first_layer))
    inputs3 = inputs1
    outputs1 = _copy_graph(split_layer, {first_layer.name: inputs1})
    outputs3.append(outputs1)
    if model_config.encoder != "None":
        encoder = encoders[model_config.encoder](**model_config.encoder_args)
        outputs1 = encoder(outputs1)
        outputs3.append(outputs1)
    x = outputs1

    # Server-side (with decoder)
    inputs2 = keras.Input(_output_shape(split_layer), dtype=outputs1.dtype)
    inputs2_ = inputs2
    if model_config.decoder != "None":
        decoder = decoders[model_config.decoder](**model_config.decoder_args)
        inputs2_ = decoder(inputs2_)
        x = decoder(x)
        outputs3.append(x)
    outputs2 = _copy_graph(layers[-1], {split_layer.name: inputs2_})
    x = _copy_graph(layers[-1], {split_layer.name: x})
    outputs3.append(x)

    model1 = keras.Model(inputs=inputs1, outputs=outputs1)
    model2 = keras.Model(inputs=inputs2, outputs=outputs2)
    model3 = keras.Model(inputs=inputs3, outputs=outputs3)

    return model1, model2, model3


def _copy_graph(layer: Layer, layer_lut: Dict[str, Tensor]) -> Tensor:
    """Recursively copy graph.

    Starting from the given layer, recursively copy graph consisting of
    all inbound layers until a layer node containing referencing a graph
    is found within the lookup table."""
    lookup = layer_lut.get(layer.name, None)
    if lookup is not None:
        return lookup

    inbound_layers = layer.inbound_nodes[0].inbound_layers

    x = (
        [_copy_graph(x, layer_lut) for x in inbound_layers]
        if isinstance(inbound_layers, list)
        else _copy_graph(inbound_layers, layer_lut)
    )

    lookup = layer(x)
    layer_lut[layer.name] = lookup
    return lookup


def _get_layer_idx_by_name(model: keras.Model, name: str) -> int:
    """Get layer index in a model by name."""
    return next(
        i for i, layer in enumerate(model.layers) if layer.name == name
    )


def _input_shape(layer: Layer) -> Tuple[int, ...]:
    """Return layer input shape, trimming first dimension."""
    return _squeeze_shape(layer.input_shape)[1:]


def _output_shape(layer: Layer) -> Tuple[int, ...]:
    """Return layer output shape, trimming first dimension."""
    return _squeeze_shape(layer.output_shape)[1:]


def _squeeze_shape(
    shape: Union[Sequence[int], Sequence[Sequence[int]]]
) -> Sequence[int]:
    """Squeeze shape into 1D."""
    if isinstance(shape[0], Sequence):
        if len(shape) > 1:
            raise ValueError("Shape can't be squeezed into 1D.")
        shape = shape[0]
    return shape
