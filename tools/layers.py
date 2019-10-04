import tensorflow as tf
import tensorflow.keras.backend as K  # pylint: disable=import-error
from tensorflow import keras
from tensorflow.python.keras.layers import Layer


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


_encoders = [UniformQuantizationU8Encoder]
_decoders = [UniformQuantizationU8Decoder]
encoders = {x.__name__: x for x in _encoders}
decoders = {x.__name__: x for x in _decoders}
