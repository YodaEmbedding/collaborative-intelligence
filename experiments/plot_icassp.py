from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

import src.analysis.dataset as ds
from src.analysis import plot
from src.analysis.experimentrunner import ExperimentRunner
from src.analysis.utils import tf_disable_eager_execution

tf_disable_eager_execution()

BATCH_SIZE = 4
DATASET_SIZE = 4
data_dir = "data"


def sample_img() -> np.ndarray:
    return np.array(Image.open(f"{data_dir}/sample/sample300.jpg"))


def rot(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])


def scale(x: float, y: float) -> np.ndarray:
    return np.array([
        [x, 0.],
        [0., y],
    ])


def skewx(x: float) -> np.ndarray:
    return np.array([
        [1., x],
        [0., 1.],
    ])


def skewy(y: float) -> np.ndarray:
    return np.array([
        [1., 0.],
        [y,  1.],
    ])


# def transform(img: np.ndarray) -> np.ndarray:
#
#     mat = np.array([
#         [1., 0., -110.],
#         [0., 1., 40.]
#     ])
#
#     # mat_tensor = mat except some rescaling (?)
#
#     mat[:2, :2] = skewx(0.1) @ rot(30/180*np.pi) @ scale(1.1, 1.3) @ mat[:2, :2]
#
#     return cv2.warpAffine(img, mat, dsize=img.shape[:-1])

# TODO interp


def main():
    runner = ExperimentRunner(
        model_name="resnet34",
        layer_name="add_3",
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )

    shape = runner.tensor_layout.shape
    dtype = runner.tensor_layout.dtype
    h, w, c = shape

    NUM_FRAMES = 2
    x_in_per_cl = 224 / w

    # TODO try with different transform parameters...

    # Define transform
    mat = np.array([
        [1., 0., -110.],
        [0., 1., 40.]
    ])

    mat[:2, :2] = skewx(0.1) @ rot(30/180*np.pi) @ scale(1.1, 1.3) @ mat[:2, :2]

    def transform(img, mat):
        return cv2.warpAffine(
            img, mat, dsize=img.shape[:-1], flags=cv2.INTER_CUBIC
        )

    def predict(ref_tensor, mat):
        mat_adj = mat.copy()
        mat_adj[:, 2] /= x_in_per_cl
        pred = np.zeros_like(ref_tensor)
        channels = ref_tensor.shape[-1]
        dsize = ref_tensor.shape[:-1]
        flags = cv2.INTER_CUBIC

        for k in range(channels):
            pred[..., k] = cv2.warpAffine(
                ref_tensor[..., k], mat_adj, dsize=dsize, flags=flags
            )

        return pred

    # Begin experiment
    img = sample_img()
    n = NUM_FRAMES
    frames = np.zeros((n, 224, 224, 3), dtype=img.dtype)
    frames[0] = img[:224, :224]
    frames[1] = transform(img, mat)[:224, :224]
    client_tensors = runner.model_client.predict(frames)
    ref_tensor = client_tensors[0]
    preds = np.zeros((n - 1, *shape), dtype=dtype)
    diffs = np.zeros((n - 1, *shape), dtype=dtype)

    # Predict
    preds[0] = predict(ref_tensor, mat)

    # Compute differences
    mask = predict(np.ones(shape, dtype=dtype), mat) != 0
    diffs = preds - client_tensors[1:]
    diffs[:, ~mask] = 0

    # Compute metrics
    mses = np.mean(diffs[:, mask] ** 2, axis=-1)
    r = np.max(client_tensors[1:]) - np.min(client_tensors[1:])
    psnrs = 10 * np.log(r ** 2 / mses)
    print(f"mse  {mses}")
    print(f"psnr {psnrs}")

    # Display results

    # Adjust for visual purposes
    tensors_ = client_tensors
    preds_ = preds
    diffs_ = np.abs(diffs)

    # Show only k channels
    koff = 13
    k = 3 ** 2
    tensors_ = tensors_[..., koff : koff + k]
    preds_ = preds_[..., koff : koff + k]
    diffs_ = diffs_[..., koff : koff + k]

    # Scale colorbar in a consistent manner
    clim = (tensors_.min(), tensors_.max())
    r = clim[1] - clim[0]
    clim_diff = (0, r)

    # Manual overrides
    clim = (-1.6, 1.4)
    clim_diff = (0, 3.0)

    # Plot
    fig = plot.featuremapsequence(
        frames,
        tensors_,
        preds_,
        diffs_,
        title="",
        clim=clim,
        clim_diff=clim_diff,
    )

    suffix = f"icassp"
    plot.save(
        fig,
        f"img/experiment/icassp/{runner.basename}-{suffix}.png",
        bbox_inches="tight",
    )

    # Why is cubic worse? Due to reduced smoothing? Or because of phase offset?

    # TODO cv2.warpPerspective

    # TODO impulse responses to discover interesting info

    # TODO Also add histogram of MSEs

    # TODO Determine a "noise model"?
    # Applications: e.g. see if low variance or non-zero mean


main()
