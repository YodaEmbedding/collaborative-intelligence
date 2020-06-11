from contextlib import suppress
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import signal
from tensorflow import keras

from src.analysis import plot
from src.analysis.methods.accuracyvskb import (
    _bin_for_plot,
    _dataframe_normalize,
    _plot_accuracy_vs_kb,
)
from src.analysis.utils import basename_of, get_cut_layers, title_of
from src.lib.split import _squeeze_shape

LEGEND = "bottom" # options: inside, right, bottom
REFRESH_SINGLE_PLOT = False
SHAPE_LABEL = True

model_name = "resnet34"
basedir = "img/accuracyvskb"
codec = "jpeg"
subdir = f"{codec}_uniquant256/{model_name}"

prefix = f"models/{model_name}/{model_name}"
model = keras.models.load_model(f"{prefix}-full.h5", compile=False)
cut_layers = [x for x in get_cut_layers(model.layers[0])]

layers = [x.name for x in cut_layers]
shapes = [_squeeze_shape(x.output_shape)[1:] for x in cut_layers]
n = len(layers)

shareddir = path.join(basedir, subdir)
filename_server = path.join(shareddir, f"{model_name}-server.csv")
data_server = pd.read_csv(filename_server)
_dataframe_normalize(data_server)

bins = np.logspace(0, np.log10(30), num=30) * 1024
data_server = _bin_for_plot(data_server, "kbs", bins / 1024)

shared_dataframes = []

for i, (layer_name, shape) in enumerate(zip(layers, shapes)):
    with suppress(FileNotFoundError):
        basename = basename_of(model_name, layer_name, i, n)
        filename_shared = path.join(shareddir, f"{basename}-shared.csv")
        data_shared = pd.read_csv(filename_shared)
        _dataframe_normalize(data_shared)
        data_shared = _bin_for_plot(data_shared, "kbs", bins / 1024)
        shape_s = r" \times ".join(map(str, shape))
        if SHAPE_LABEL:
            label = f"({i + 1} / {n}) (${shape_s}$): {layer_name}"
        else:
            label = f"({i + 1} / {n}): {layer_name}"
        shared_dataframes.append((label, data_shared))
        if REFRESH_SINGLE_PLOT:
            title = title_of(model_name, layer_name, i, n)
            fig = _plot_accuracy_vs_kb(title, data_server, data_shared)
            plot.save(fig, path.join(shareddir, f"{basename}.png"))
        print(label)


max_accuracy = 0.806
thresh_pct = 0.95
thresh = thresh_pct * max_accuracy


def process_dataframe(df: pd.DataFrame):
    df = df.groupby("kbs").agg({"acc": "mean"}).reset_index()
    y = df["acc"]
    y = signal.savgol_filter(y, 5, 2)
    y = signal.savgol_filter(y, 5, 1)
    df["acc"] = y
    return df


dfs = [("server-only inference", data_server)] + shared_dataframes[1:]
dfs = [(label, process_dataframe(df)) for label, df in dfs]


max_height = 30.0
heights = []
xlabels = []
for label, df in dfs:
    idx = next((i for i, y_ in enumerate(df["acc"]) if y_ >= thresh), None)
    height = max_height
    if idx is not None:
        if idx == 0:
            height = df["kbs"].iloc[idx]
        else:
            x0 = df["kbs"].iloc[idx - 1]
            x1 = df["kbs"].iloc[idx]
            y0 = df["acc"].iloc[idx - 1]
            y1 = df["acc"].iloc[idx]
            height = (thresh - y0) * (x1 - x0) / (y1 - y0) + x0
    xs = label.split(":")
    xlabel = xs[-1][1:] if len(xs) > 1 else xs[-1]
    heights.append(height)
    xlabels.append(xlabel)

heights.extend([0, 0, 0])
xlabels.extend(["pool1", "fc1", "softmax"])

fig = plot.model_bar(
    heights, xlabels, model_name, "Compressed output tensor size (KiB)"
)
ax, = fig.axes
ax.axhline(y=heights[0], color=(0.7, 0.7, 1.0), linestyle=":")
plot.save(fig, path.join(shareddir, f"{model_name}-compressed.png"))


if LEGEND == "inside":
    fig = plt.figure()
elif LEGEND == "right":
    fig = plt.figure(figsize=(12.5, 6))
elif LEGEND == "bottom":
    fig = plt.figure(figsize=(8, 10))
n_colors = len(dfs)
pal = sns.cubehelix_palette(start=2, rot=1, reverse=True, n_colors=n_colors)
sns.set_palette(pal)
for label, df in dfs[:1]:
    ax = sns.lineplot(x="kbs", y="acc", data=df, label=label, ci=0)
    ax.lines[0].set_linestyle("--")
    ax.lines[0].set_zorder(100)
for label, df in dfs[1:]:
    sns.lineplot(x="kbs", y="acc", data=df, label=label, ci=0)
ax: plt.Axes = plt.gca()
ax.legend().remove()
ax.set(xlim=(0, 30), ylim=(0, 1))
ax.set_xlabel("Compressed size (KB)")
ax.set_ylabel("Accuracy")

if LEGEND == "inside":
    fig.tight_layout()
    ax.legend(loc="lower right", ncol=2, prop={"size": 5}, framealpha=0.95)
elif LEGEND == "right":
    fig.tight_layout()
    fig.legend(loc=7, ncol=2)
    fig.subplots_adjust(right=0.55)
elif LEGEND == "bottom":
    fig.tight_layout()
    fig.legend(loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.45)

plot.save(fig, path.join(shareddir, f"{model_name}-aggregate.png"))
