from time import time

import numpy as np
from tensorflow import keras

from src.analysis.dataset import single_sample_image
from src.analysis.utils import compile_kwargs, release_models
from src.lib.split import split_model


# @separate_process()
def analyze_latencies_layer(model: keras.Model, layer_name: str):
    model_client, model_server, model_analysis = split_model(
        model, layer=layer_name
    )
    model_client.compile(**compile_kwargs)
    model_server.compile(**compile_kwargs)
    model_analysis.compile(**compile_kwargs)
    num_samples = 100
    data = single_sample_image()[np.newaxis].astype(np.float32)
    t0 = time()
    for _ in range(num_samples):
        model_client.predict(data)
    t1 = time()
    t = (t1 - t0) / num_samples
    release_models(model_client, model_server, model_analysis)
    print(f"{layer_name:20}{int(t * 1000)}")
