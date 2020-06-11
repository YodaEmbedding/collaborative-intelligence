import numpy as np
import tensorflow as tf

from src.analysis.methods.featuremap import analyze_featuremapcompression_layer
from src.analysis.methods.histograms import analyze_histograms_layer
from src.analysis.utils import basename_of, compile_kwargs, title_of
from src.lib.layers import (
    UniformQuantizationU8Decoder,
    UniformQuantizationU8Encoder,
)
from src.lib.split import split_model
from src.analysis.quant import uni_quant

model_name = "resnet34"
layer_name = "add_3"
i = 15
n = 37
kbs = [3, 5, 10, 30]
title = title_of(model_name, layer_name, i, n)
basename = basename_of(model_name, layer_name, i, n)

model = tf.keras.models.load_model(
    f"models/{model_name}/{model_name}-full.h5", compile=False
)

model_client, model_server, model_analysis = split_model(
    model, layer=layer_name
)
model_client.compile(**compile_kwargs)
model_server.compile(**compile_kwargs)

d = {}
d.update(analyze_histograms_layer(model_client, title, basename))
sigma = 3
clip_range = (d["mean"] - sigma * d["std"], d["mean"] + sigma * d["std"])

model_client_u8, model_server_u8, model_analysis_u8 = split_model(
    model,
    layer=layer_name,
    encoder=UniformQuantizationU8Encoder(clip_range),
    decoder=UniformQuantizationU8Decoder(clip_range),
)

# postprocess = lambda x: x
# postprocess = lambda x: uni_quant(x, (32, 255 - 32), 256)
# postprocess = lambda x: uni_quant(x.astype(np.float), (0, 255), 256)
# postprocess = lambda x: uni_quant(x.astype(np.float), (32, 255 - 32), 256)
quant = lambda x: x
analyze_featuremapcompression_layer(
    model_client_u8, title, basename, quant, kbs  # , postprocess=postprocess
)

# TODO measure if better MSE from sigma=4 or sigma=3
