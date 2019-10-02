from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow.keras.backend as K  # pylint: disable=import-error
from tensorflow import keras
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.layers import Layer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@dataclass
class SplitOptions:
    layer: str
    encoder: Callable[[Tensor], Tensor]
    decoder: Callable[[Tensor], Tensor]

    def __str__(self):
        return f"{self.layer}-{self.encoder}-{self.decoder}"


class UniformQuantizationU8Encoder(Layer):
    """Client-side encoding."""

    def __init__(self, clip_range, **kwargs):
        self.clip_range = clip_range
        self._scale = 255 / (self.clip_range[1] - self.clip_range[0])
        super(UniformQuantizationU8Encoder, self).__init__(**kwargs)

    def __str__(self):
        return f"{type(self).__name__}(clip_range={tuple(self.clip_range)})"

    def call(self, inputs, **kwargs):
        x = inputs
        # x = K.log(x)
        # x = K.clip(x, -4, 1)
        # x = (x + 4) * (255 / 5)
        x = (x - self.clip_range[0]) * self._scale
        x = K.cast(x, "uint8")
        return x

    def get_config(self):
        config = {"clip_range": self.clip_range}
        config.update(super(UniformQuantizationU8Encoder, self).get_config())
        return config


class UniformQuantizationU8Decoder(Layer):
    """Server-side decoding."""

    def __init__(self, clip_range, **kwargs):
        self.clip_range = clip_range
        self._scale = (self.clip_range[1] - self.clip_range[0]) / 255
        super(UniformQuantizationU8Decoder, self).__init__(**kwargs)

    def __str__(self):
        return f"{type(self).__name__}(clip_range={tuple(self.clip_range)})"

    def call(self, inputs, **kwargs):
        x = inputs
        x = K.cast(x, "float32")
        x = x * self._scale + self.clip_range[0]
        return x

    def get_config(self):
        config = {"clip_range": self.clip_range}
        config.update(super(UniformQuantizationU8Decoder, self).get_config())
        return config


def split_model(
    model: keras.Model, split_options: SplitOptions
) -> Tuple[keras.Model, keras.Model]:
    """Split model by given layer index.

    Attaches encoder layer to end of client model. Attaches decoder
    layer to beginning of server model.
    """
    split_idx = _get_layer_idx_by_name(model, split_options.layer)
    layers = model.layers
    first_layer = layers[0]
    split_layer = layers[split_idx]

    inputs1 = keras.Input(_input_shape(first_layer))
    outputs1 = _copy_graph(split_layer, {first_layer.name: inputs1})
    if split_options.encoder is not None:
        outputs1 = split_options.encoder(outputs1)
    model1 = keras.Model(inputs=inputs1, outputs=outputs1)

    inputs2 = keras.Input(_output_shape(split_layer), dtype=outputs1.dtype)
    inputs2_ = inputs2
    if split_options.decoder is not None:
        inputs2_ = split_options.decoder(inputs2_)
    outputs2 = _copy_graph(layers[-1], {split_layer.name: inputs2_})
    model2 = keras.Model(inputs=inputs2, outputs=outputs2)

    return model1, model2


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
