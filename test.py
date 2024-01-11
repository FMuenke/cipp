import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from data_structure.segmentation_data_set import SegmentationDataSet
from data_structure.folder import Folder
from conventional_image_processing_pipeline.model import Model
from utils.utils import load_dict
from data_structure.model_statistics import StatsHandler


def convert_cls_to_color(cls_map, color_coding, unsupervised=False):
    h, w = cls_map.shape[:2]
    color_map = np.zeros((h, w, 3))
    if unsupervised:
        unique_y = np.unique(cls_map)
        for u in unique_y:
            if str(u) not in color_coding:
                color_coding[str(u)] = [[0, 0, 0],
                                        [np.random.randint(255), np.random.randint(255), np.random.randint(255)]]
    for idx, cls in enumerate(color_coding):
        iy, ix = np.where(cls_map == idx + 1)
        color_map[iy, ix, :] = [color_coding[cls][1][2],
                                color_coding[cls][1][1],
                                color_coding[cls][1][0]]
    return color_map


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    us = args_.unsupervised

    color_coding = load_dict(os.path.join(mf, "color_coding.json"))

    model = Model(mf)
    model.load(mf)

    d_set = SegmentationDataSet(df, color_coding)
    t_set = d_set.load()
    tags, _ = d_set.split(t_set, 0.0)
    model.evaluate(tags, color_coding, mf)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--model_folder", "-model", default="./test", help="Path to model directory"
    )
    parser.add_argument(
        "--unsupervised", "-unsup", type=bool, default=False, help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
