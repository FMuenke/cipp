import os
import cv2
import numpy as np
from PIL import Image


def convert(path):
    labels_path = os.path.join(path, "labels")
    dst_path = os.path.join(path, "label_new")
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    for i in os.listdir(labels_path):
        if i.endswith(".tif"):
            print(os.path.join(labels_path, i))
            img = Image.open(os.path.join(labels_path, i))
            img = np.array(img)
            print(np.max(img))
            img[img > 0] = 255
            print(os.path.join(dst_path, i.replace(".tif", ".png")), np.unique(img))
            cv2.imwrite(os.path.join(dst_path, i.replace(".tif", ".png")), img.astype(np.uint8))


def main():
    path = "/Users/fmuenke/datasets/2021_live_cell_SkBr3_cropped/"

    convert(os.path.join(path, "train"))
    convert(os.path.join(path, "test"))



if __name__ == "__main__":
    main()
