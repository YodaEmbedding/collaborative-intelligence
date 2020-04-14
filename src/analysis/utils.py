import gc
from functools import wraps
from multiprocessing import Process, Queue
from time import sleep
from typing import Iterator, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

from src.modelconfig import ModelConfig

compile_kwargs = {
    "loss": "sparse_categorical_crossentropy",
    "optimizer": keras.optimizers.RMSprop(),
    "metrics": [],
    "run_eagerly": False,
}


def get_cut_layers(root: Layer) -> Iterator[Layer]:
    """Obtain list of all layers that cut the graph.

    Given the root layer of a directed acyclic graph (DAG) with a single
    input layer and single output layer, yield root layer, then yield
    any layers that are "cut vertices" in order of distance from the
    root layer, and finally, yield output layer.
    """
    fringe = {root}
    waiting = {}
    visited = set()

    while len(fringe) != 0:
        curr = fringe.pop()
        visited.add(curr)

        if len(fringe) + len(waiting) == 0:
            yield curr

        for node in curr.outbound_nodes:
            node = node.outbound_layer
            x = node.inbound_nodes[0].inbound_layers
            inbound_layers = x if isinstance(x, list) else [x]
            if node in visited:
                continue
            if node in waiting:
                waiting[node] -= 1
                if waiting[node] == 0:
                    del waiting[node]
                    fringe.add(node)
            elif len(inbound_layers) > 1:
                waiting[node] = len(inbound_layers) - 1
            else:
                fringe.add(node)


def basename_of(
    model_name: str, layer_name: str, layer_i: int, layer_n: int
) -> str:
    d = len(str(layer_n))
    return f"{model_name}-{layer_i+1:0{d}}of{layer_n:0{d}}-{layer_name}"


def prefix_of(model_config: ModelConfig) -> str:
    return f"models/{model_config.to_path()}"


def title_of(
    model_name: str, layer_name: str, layer_i: int, layer_n: int
) -> str:
    return f"{model_name} {layer_name} ({layer_i+1}/{layer_n})"


def release_models(*models: List[keras.Model]):
    for model in models:
        del model
        gc.collect()
    # gc.collect()
    # keras.backend.clear_session()
    # tf.compat.v1.reset_default_graph()
    # gc.collect()


def tf_disable_eager_execution():
    """Disable eager execution globally and force graph mode."""
    from tensorflow.python.framework.ops import disable_eager_execution

    disable_eager_execution()
    # tf.compat.v1.disable_eager_execution()


def tf_gpu_grow_memory():
    """Only allocate GPU memory when needed."""
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)


def new_tf_graph_and_session(func):
    """Run decorated function in a new tf.Graph and tf.Session process."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        graph = tf.Graph()
        sess = tf.compat.v1.Session(graph=graph)
        with sess:
            with graph.as_default():
                return func(*args, **kwargs)

    return wrapper


def separate_process(sleep_after: int = 0):
    """Run decorated function in a separate process."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            q = Queue()

            def worker(q, *args, **kwargs):
                q.put(func(*args, **kwargs))

            p = Process(target=worker, args=(q, *args), kwargs=kwargs)
            p.start()
            result = q.get()
            p.join()
            sleep(sleep_after)
            return result

        return wrapper

    return decorator
