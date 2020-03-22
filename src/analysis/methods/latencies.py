from time import time

import numpy as np
from tensorflow import keras

from src.analysis.dataset import single_sample_image


# TODO plot bar chart
# TODO exclude GPU -> RAM data copy time (e.g. via Stage or tf.profiler)
# TODO plot stacked bar chart including GPU -> RAM data copy time
def analyze_latencies_layer(model_client: keras.Model, layer_name: str):
    num_samples = 100
    data = single_sample_image()[np.newaxis].astype(np.float32)
    t0 = time()
    for _ in range(num_samples):
        model_client.predict(data)
    t1 = time()
    t = (t1 - t0) / num_samples
    print("{}".format("cumulative time (ms)"))
    print(f"{int(t * 1000)}")
