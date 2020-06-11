import json

import numpy as np
import skimage.measure
import tensorflow as tf
from scipy import stats

from src.analysis import dataset as ds
from src.analysis import plot
from src.analysis.experimentrunner import ExperimentRunner
from src.analysis.quant import uni_dequant, uni_quant
from src.analysis.utils import (
    categorical_top1_accuracy,
    tf_disable_eager_execution,
)
from src.lib.layouts import TensorLayout
from src.lib.postencode import _pil_encode
from src.lib.predecode import _decode_raw_img
from src.lib.tile import determine_tile_layout, tile

tf_disable_eager_execution()

with open("config.json") as f:
    config = json.load(f)

BATCH_SIZE = config["batch_size"]
DATASET_SIZE = config["dataset_size"]
TEST_DATASET_SIZE = config["test_dataset_size"]


def main():
    runner = ExperimentRunner(
        model_name="resnet34",
        layer_name="add_3",
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        test_dataset_size=TEST_DATASET_SIZE,
    )

    model_name = runner.model_name
    basename = runner.basename
    title = runner.title
    tensor_layout = runner.tensor_layout
    # basename_stats = f"img/stats/{model_name}/{basename}"
    h, w, c = tensor_layout.shape_in_order("hwc")

    x_tensors = runner.data.client_tensors
    xs = x_tensors.reshape(len(x_tensors), -1)
    ps = np.zeros(h * w * c)

    for i in range(h * w * c):
        _w_stat, p_value = stats.shapiro(xs[:, i])
        ps[i] = p_value

    ps = ps.reshape((h, w, c))
    ps = np.clip(ps, 0, 0.05)

    nonnormal = x_tensors[:, ps < 0.01]
    normal = x_tensors[:, ps >= 0.05]
    # nonnormal = x_tensors[:, np.argmin(ps)]

    def plot_quick(plot_func, tensor, suffix, **kwargs):
        filename = f"img/experiment/{basename}-{suffix}.png"
        fig = plot_func(tensor, runner.title, **kwargs)
        plot.save(fig, filename, bbox_inches="tight")

    plot_quick(plot.featuremap, ps, "normality")
    plot_quick(plot.neuron_histogram, nonnormal[:, 0], "nonnormal", bins=40)
    plot_quick(plot.neuron_histogram, normal[:, 0], "normal", bins=40)
    plot_quick(
        plot.neuron_histogram,
        xs[:, np.argmin(ps)],
        "reallynonnormal",
        bins=40,
    )
    plot_quick(
        plot.neuron_histogram, xs[:, np.argmax(ps)], "reallynormal", bins=40
    )


main()
