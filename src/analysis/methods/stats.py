import numpy as np

from src.analysis import plot
from src.analysis.experimentrunner import ExperimentRunner
from src.analysis.utils import categorical_top1_accuracy


def analyze_stats_layer(runner: ExperimentRunner):
    basename = f"img/stats/{runner.model_name}/{runner.basename}"

    def plot_quick(plot_func, tensor, suffix, **kwargs):
        fig = plot_func(tensor, runner.title, **kwargs)
        plot.save(fig, f"{basename}-{suffix}.png", bbox_inches="tight")

    def load(suffix, f, *args, **kwargs):
        filename = f"{basename}-{suffix}.npy"
        try:
            return np.load(filename)
        except FileNotFoundError:
            arr = f(*args, **kwargs)
            np.save(filename, arr)
            return arr

    client_tensors = runner.data.client_tensors

    d = {}
    d["tensors"] = client_tensors
    d["tensors_mean"] = load("tensors_mean", np.mean, client_tensors, axis=0)
    d["tensors_std"] = load("tensors_std", np.std, client_tensors, axis=0)
    d["tensors_min"] = load("tensors_min", np.min, client_tensors, axis=0)
    d["tensors_max"] = load("tensors_max", np.max, client_tensors, axis=0)
    d["mean"] = np.mean(client_tensors)
    d["std"] = np.std(client_tensors)
    pred_tensors = runner.model_server.predict(client_tensors)
    d["pred_tensors"] = pred_tensors
    labels = runner.data.labels
    d["accuracy"] = np.mean(categorical_top1_accuracy(labels, pred_tensors))
    runner.d.update(d)

    plot_quick(plot.featuremap, d["tensors_mean"], "mean")
    plot_quick(plot.featuremap, d["tensors_std"], "std")
    plot_quick(plot.featuremap, d["tensors_min"], "min")
    plot_quick(plot.featuremap, d["tensors_max"], "max")
