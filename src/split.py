from typing import Dict, List, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.python.framework.ops import Tensor

from .layers import decoders, encoders
from .modelconfig import ModelConfig

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def split_model(
    model: keras.Model, model_config: ModelConfig
) -> Tuple[keras.Model, keras.Model, keras.Model]:
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
    last_layer = layers[-1]
    x_dict = {}

    # NOTE
    # model1 == model_client
    # model2 == model_server
    # model3 == model_analysis

    # Client-side (input)
    x = keras.Input(_input_shape(first_layer))
    inputs3 = inputs1 = x

    # Client-side (first half of model)
    x = _copy_graph(split_layer, {first_layer.name: x})
    x_dict["client_result"] = x

    # Client-side (encoder)
    if model_config.encoder != "None":
        encoder = encoders[model_config.encoder](**model_config.encoder_args)
        x = encoder(x)
        x_dict["client_encoded"] = x

    # Client-side (final client-side output)
    outputs1 = x

    # Server-side (input)
    # Create new graph for model_server, bound to x2
    x2 = keras.Input(_output_shape(split_layer), dtype=x.dtype)
    inputs2 = x2

    # Server-side (decoder)
    if model_config.decoder != "None":
        decoder = decoders[model_config.decoder](**model_config.decoder_args)
        x = decoder(x)
        x2 = decoder(x2)
        x_dict["server_decoded"] = x

    # Server-side (second half of model)
    x = _copy_graph(last_layer, {split_layer.name: x})
    x2 = _copy_graph(last_layer, {split_layer.name: x2})
    x_dict["server_result"] = x

    # Server-side (final server-side output)
    outputs2 = x2

    # Analysis model outputs
    outputs3 = _analysis_outputs(x_dict)

    model1 = keras.Model(inputs=inputs1, outputs=outputs1)
    model2 = keras.Model(inputs=inputs2, outputs=outputs2)
    model3 = keras.Model(inputs=inputs3, outputs=outputs3)

    return model1, model2, model3


def copy_model(model: keras.Model) -> Tuple[keras.Model, keras.Model]:
    # return keras.models.clone_model(model)
    layers = model.layers
    first_layer = layers[0]
    last_layer = layers[-1]
    inputs = keras.Input(_input_shape(first_layer))
    outputs = _copy_graph(last_layer, {first_layer.name: inputs})
    return keras.Model(inputs=inputs, outputs=outputs)


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


def _analysis_outputs(d: Dict[str, Tensor]) -> List[Tensor]:
    outputs = []

    outputs.append(d["client_result"])

    if "client_encoded" in d:
        outputs.append(d["client_encoded"])

    if "server_decoded" in d:
        outputs.append(d["server_decoded"])

    outputs.append(d["server_result"])

    return outputs


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
