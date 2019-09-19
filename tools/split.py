from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras
# pylint: disable-msg=E0611
from tensorflow.python.keras.layers import Layer
import tensorflow.python.keras.backend as K
#pylint: enable-msg=E0611

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class EncoderLayer(Layer):
    """Client-side encoding."""

    def __init__(self, clip_range, **kwargs):
        self.clip_range = clip_range
        super(EncoderLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        # x = K.log(x)
        # x = K.clip(x, -4, 1)
        # x = (x + 4) * (255 / 5)
        scale = 255 / (self.clip_range[1] - self.clip_range[0])
        x = (x - self.clip_range[0]) * scale
        x = K.cast(x, 'uint8')
        return x

    def get_config(self):
        config = {'clip_range': self.clip_range}
        config.update(super(EncoderLayer, self).get_config())
        return config

class DecoderLayer(Layer):
    """Server-side decoding."""

    def __init__(self, clip_range, **kwargs):
        self.clip_range = clip_range
        super(DecoderLayer, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        scale = (self.clip_range[1] - self.clip_range[0]) / 255
        x = K.cast(x, 'float32')
        x = x * scale + self.clip_range[0]
        return x

    def get_config(self):
        config = {'clip_range': self.clip_range}
        config.update(super(DecoderLayer, self).get_config())
        return config

def split_model(
    model: keras.Model,
    split_layer_name: str
) -> Tuple[keras.Model, keras.Model]:
    """Split model by given layer index.

    Attaches encoder layer to end of client model. Attaches decoder
    layer to beginning of server model.
    """
    split_idx = _get_layer_idx_by_name(model, split_layer_name)
    clip_range = (-2, 2)
    layers = model.layers
    first_layer = layers[0]
    split_layer = layers[split_idx]

    input_layer1 = keras.Input(_input_shape(first_layer))
    outputs1 = _copy_graph(split_layer, {first_layer.name: input_layer1})
    outputs1 = EncoderLayer(clip_range)(outputs1)
    model1 = keras.Model(inputs=input_layer1, outputs=outputs1)
    # model1 = _model_from_layers(split_layer, layers[0])

    input_layer2 = keras.Input(_input_shape(split_layer), dtype='uint8')
    top_layer2 = DecoderLayer(clip_range)(input_layer2)
    outputs2 = _copy_graph(layers[-1], {split_layer.name: top_layer2})
    model2 = keras.Model(inputs=input_layer2, outputs=outputs2)
    # model2 = _model_from_layers(layers[-1], split_layer)

    return model1, model2

def _copy_graph(layer: Layer, layer_lut: Dict[Layer, Layer]) -> Layer:
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
        if isinstance(inbound_layers, list) else
        _copy_graph(inbound_layers, layer_lut))

    lookup = layer(x)
    layer_lut[layer.name] = lookup
    return lookup

# def _model_from_layers(layer, top_layer) -> keras.Model:
#     shape = top_layer.input_shape
#     shape = shape[0][1:] if isinstance(shape, list) else shape[1:]
#     input_layer = keras.Input(shape)
#     outputs = _copy_graph(layer, {top_layer.name: input_layer})
#     return keras.Model(inputs=input_layer, outputs=outputs)

def _get_layer_idx_by_name(model: keras.Model, name: str) -> int:
    return next(i for i, layer in enumerate(model.layers) if layer.name == name)

def _input_shape(layer: Layer) -> Tuple[int, ...]:
    """Determine layer input shape."""
    shape = layer.input_shape
    return shape[0][1:] if isinstance(shape, list) else shape[1:]

def _model_from_layers(layer, top_layer) -> keras.Model:
    """Create model from subgraph between `layer` and `top_layer`."""
    input_layer = keras.Input(_input_shape(top_layer))
    outputs = _copy_graph(layer, {top_layer.name: input_layer})
    return keras.Model(inputs=input_layer, outputs=outputs)
