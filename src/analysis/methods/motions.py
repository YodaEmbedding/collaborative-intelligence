import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from src.analysis import plot
from src.analysis.utils import basename_of, title_of
from src.lib.layouts import TensorLayout
from src.lib.tile import as_order, determine_tile_layout, tile_chw


def analyze_motions_layer(
    model_name: str,
    model_client: keras.Model,
    layer_name: str,
    layer_i: int,
    layer_n: int,
):
    shape = model_client.output_shape[1:]
    if len(shape) != 3:
        return
    h, w, c = shape

    # frames = np.random.randint(256, size=(64, 224, 224, 3)).astype(np.uint8)
    img = np.array(Image.open("sampleWx224.jpg"))
    speed = 10.0
    num_frames = int((img.shape[1] - 224) / speed)
    frames = np.empty((num_frames, 224, 224, 3), dtype=img.dtype)
    for i in range(num_frames):
        x = int(speed * i)
        frames[i] = img[:, x : x + 224]

    tensors = model_client.predict(frames.astype(np.float32))
    tensors = as_order(tensors, "hwc", "chw")
    num_flows = len(tensors) - 1
    flows = np.empty((num_flows, c, h, w, 2))

    # TODO does this optical flow experiment really provide any useful
    # information? Maybe only for translation, but for *noise*??? What were you
    # expecting?!

    # Some sort of... decomposition between "static" neurons and "dynamic"
    # neuron outputs?

    for i, (tensor, tensor_next) in enumerate(zip(tensors[1:], tensors[:-1])):
        for k in range(c):
            flow = cv2.calcOpticalFlowFarneback(
                tensor[k],
                tensor_next[k],
                flow=None,
                pyr_scale=0.5,
                levels=3,
                iterations=3,
                winsize=15,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            # r, th = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # flows[i, k, ..., 0] = r
            # flows[i, k, ..., 1] = th
            flows[i, k] = flow

    tensor_layout = TensorLayout(np.float64, c, h, w, "chw")
    tiled_layout = determine_tile_layout(tensor_layout)
    ncols = tiled_layout.ncols
    nrows = tiled_layout.nrows
    flow_frames = np.empty((num_flows, nrows * h, ncols * w, 2))
    tensor_frames = np.empty((num_flows, nrows * h, ncols * w))

    for i in range(num_flows):
        tensor_frames[i] = tile_chw(tensors[i], nrows, ncols)
        flow_frames[i] = tile_chw(flows[i], nrows, ncols)

    title = title_of(model_name, layer_name, layer_i, layer_n)
    basename = basename_of(model_name, layer_name, layer_i, layer_n)
    ani = plot.OpticalFlowAnimator(
        frames[:-1], tensor_frames, flow_frames, title
    )
    ani.save_img(f"img/motion/{basename}.png")
    ani.save(f"img/motion/{basename}.mp4")

    print("Analyzed motion")

    # TODO output png containing sequence of images with:
    # top row: input image frame
    # med row: intermediate tensor output
    # bot row: optical flow
