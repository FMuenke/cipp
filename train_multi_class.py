import argparse
import os

from data_structure import SegmentationDataSet
from conventional_image_processing_pipeline import Model

from conventional_image_processing_pipeline import InputLayer
from conventional_image_processing_pipeline import CIPPLayer

from conventional_image_processing_pipeline import augment_data_set, Augmentations

from utils.utils import save_dict


class MultiClassModel:
    def __init__(self, base_model, color_coding, model_path):
        self.color_coding = color_coding
        self.



def main(args_):
    color_coding = {
        "crack": [[255, 255, 255], [255, 0, 0]],
        # "shadow": [[1, 1, 1], [0, 100, 255]]
        # "heart": [[4, 4, 4], [0, 100, 255]]
        # "nuceli": [[255, 255, 255], [100, 100, 255]],
    }

    randomized_split = True
    train_test_ratio = 0.10

    df = args_.dataset_folder
    mf = args_.model_folder

    x = InputLayer("IN", features_to_use="gray-color", initial_down_scale=1)
    x = CIPPLayer(x, "CIPP", operations=[
        "blurring",
        "invert",
        ["watershed", "threshold_otsu", "threshold"],
        "remove_small_holes",
        "crop",
    ], selected_layer=[0], optimizer="grid_search", use_multiprocessing=True)

    model = Model(graph=x)

    d_set = SegmentationDataSet(os.path.join(df, "train"), color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set, percentage=train_test_ratio, random=randomized_split)

    model.fit(train_set[:16], validation_set)

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
