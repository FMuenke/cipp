import argparse
import os

from data_structure import SegmentationDataSet
from conventional_image_processing_pipeline import Model

from conventional_image_processing_pipeline import InputLayer
from conventional_image_processing_pipeline import CIPPLayer

from conventional_image_processing_pipeline import augment_data_set, Augmentations

from utils.utils import save_dict


def model_v4():
    x = InputLayer("IN", features_to_use=["RGB-color"], initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "threshold_percentile",
        "fill_contours",
        "closing",
        "remove_small_objects",
    ], selected_layer=[0, 1, 2], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)

    x = InputLayer("IN", features_to_use=["RGB-color"], initial_down_scale=1)
    x = CIPPLayer(x, "SIMPLE", operations=[
        "watershed",
        "closing",
        "opening",
        "remove_small_objects",
    ], selected_layer=[1], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


def main(args_):
    color_coding = {
        "crack": [[0, 255, 0], [0, 255, 0]],
        # "pothole": [[255, 0, 0], [0, 255, 0]],
    }

    randomized_split = True
    train_test_ratio = 0.1

    df = args_.dataset_folder
    mf = args_.model_folder

    x = InputLayer("IN", features_to_use="gray-color", width=256, height=256)
    x = CIPPLayer(x, "CIPP", operations=[
        "blurring",
        "top_clipping_percentile",
        "invert",
        ["watershed", "threshold_otsu", "threshold"],
        "remove_small_objects",
    ], selected_layer=[0], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)

    d_set = SegmentationDataSet(os.path.join(df, "train"), color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    # augmentations = Augmentations(True, True, True)
    # train_set = augment_data_set(train_set, augmentations, multiplier=3)

    model.fit(train_set, validation_set)

    d_set_test = SegmentationDataSet(os.path.join(df, "test"), color_coding)
    tag_set_test = d_set_test.load()
    test_set, _ = d_set.split(tag_set_test, percentage=train_test_ratio, random=0.0)

    model.evaluate(test_set, color_coding, mf)
    model.save(mf)
    save_dict(color_coding, os.path.join(mf, "color_coding.json"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder",
        "-model",
        default="./test",
        help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
