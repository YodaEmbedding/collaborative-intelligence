from time import time
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow import keras

from src.analysis import plot
from src.analysis.dataset import single_sample_image


# TODO plot bar chart
# TODO exclude GPU -> RAM data copy time (e.g. via Stage or tf.profiler)
# TODO plot stacked bar chart including GPU -> RAM data copy time
def analyze_latencies_layer(model_client: keras.Model, layer_name: str):
    num_warmup = 1000
    num_samples = 1000
    sess = K.get_session()
    data = single_sample_image()[np.newaxis].astype(np.float32)
    data = tf.constant(data)
    out_shape = model_client.output.shape[1:]
    output_store = tf.Variable(tf.zeros((1, *out_shape)), name="output_store")
    infer_op = output_store.assign(model_client(data))

    for _ in range(num_warmup):
        _ = sess.run([infer_op])

    t0 = time()
    for _ in range(num_samples):
        _ = sess.run([infer_op])
    t1 = time()
    t = (t1 - t0) / num_samples

    print("{}".format("cumulative time (ms)"))
    print(f"{(t * 1000):.2f}")

    return t


def analyze_latencies_post(model_name: str, dicts: List[Dict[str, Any]]):
    xlabels = [d["layer"] for d in dicts]
    heights = [d["latency"] * 1000 for d in dicts]
    title = f"{model_name}"
    basename = f"{model_name}"
    fig = plot.model_bar(heights, xlabels, title, "Cumulative latency (ms)")
    plot.save(fig, f"img/latencies/{basename}.png")
