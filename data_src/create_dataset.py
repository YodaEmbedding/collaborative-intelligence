import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def process(img: np.ndarray):
    if len(img.shape) == 2:
        return

    h, w, c = img.shape

    if c != 3 or h < 224 or w < 224:
        return

    if h < w:
        x = (w - h) // 2
        img = img[:, x : x + h]
    elif h > w:
        y = (h - w) // 2
        img = img[y : y + w]

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    return img


out_data_dir = "data/uncompressed"
# out_data_dir = "data/lossless"

# data_dir = "/mnt/data/datasets/ILSVRC2012/ILSVRC2012_img_val"
# filenames = sorted(os.listdir(data_dir))

data_dir = "/mnt/data/datasets/ILSVRC2012/ILSVRC2012_img_train"
# csv_path = "/mnt/data/datasets/all-11crop-224x224/data_kb_all.csv"
csv_path = "data/data_kb_all.csv"
df = pd.read_csv(csv_path)
df = df[["file", "label"]]

filenames = [f"{label}/{filename}" for i, filename, label in df.itertuples()]
filenames.sort()

for filename in filenames:
    print(filename)
    path = os.path.join(data_dir, filename)
    with open(path, "rb") as f:
        img = np.array(Image.open(f))
    img = process(img)
    if img is None:
        continue
    img = Image.fromarray(img)
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    path = os.path.join(out_data_dir, f"{base_filename}.png")
    with open(path, "wb") as f:
        img.save(f, format="png")
