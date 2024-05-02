import argparse
import os
from time import time
import numpy as np
from tqdm import tqdm

from data_structure.segmentation_data_set import SegmentationDataSet
from conventional_image_processing_pipeline.model import Model
from data_structure.model_statistics import ModelStatistics
from data_structure.folder import Folder

from conventional_image_processing_pipeline import InputLayer
from conventional_image_processing_pipeline import CIPPLayer

from utils.utils import save_dict, load_dict


def model_cipp():
    x = InputLayer("IN", features_to_use="gray-color", height=256, width=256)
    x = CIPPLayer(x, "CIPP", operations=[
        "blurring",
        "top_clipping_percentile",
        "invert",
        ["threshold", "edge"],
        "remove_small_objects",
        "remove_small_holes"
    ], selected_layer=[0], optimizer="grid_search", use_multiprocessing=True)
    model = Model(graph=x)
    return model


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


def run_training(df, mf, number_of_tags, model, color_coding):
    print(mf)

    d_set = SegmentationDataSet(df, color_coding)
    tag_set = d_set.load()

    t0 = time()
    if number_of_tags != 0:
        tags = [tag_set[t] for t in tag_set]
        seed_id = int(mf.split("-RUN-")[-1])
        rng = np.random.default_rng(seed_id)
        tags = rng.choice(tags, number_of_tags, replace=False)
        n_val = int(max(1, number_of_tags * 0.2))
        train_set = tags[:-n_val]
        validation_set = tags[-n_val:]
        print("Number of Training images reduced! - {}/{} -".format(
            len(train_set), len(validation_set)))
        model.fit(train_set, validation_set)
        model.save(mf)
    else:
        randomized_split = True
        train_test_ratio = 0.5
        train_set, validation_set = d_set.split(
            tag_set, percentage=train_test_ratio, random=randomized_split)
        model.fit(train_set, validation_set)
        model.save(mf)
    save_dict({"train_time": time() - t0}, os.path.join(mf, "train_time_log.json"))

    save_dict(color_coding, os.path.join(mf, "color_coding.json"))


def run_test(df, mf, us):
    color_coding = load_dict(os.path.join(mf, "color_coding.json"))

    model = Model(mf)
    model.load(mf)

    res_fol = Folder(os.path.join(mf, "segmentations"))
    res_fol.check_n_make_dir(clean=True)

    vis_fol = Folder(os.path.join(mf, "overlays"))
    vis_fol.check_n_make_dir(clean=True)

    d_set = SegmentationDataSet(df, color_coding)
    t_set = d_set.load()

    sh = ModelStatistics(color_coding)
    print("Processing Images...")
    for tid in tqdm(t_set):
        cls_map = model.predict(t_set[tid].load_x())
        color_map = convert_cls_to_color(cls_map, color_coding, unsupervised=us)
        # t_set[tid].write_result(res_fol.path(), color_map)
        if not us:
            t_set[tid].eval(color_map, sh)
        if mf.endswith("RUN-XXXX"):
            t_set[tid].visualize_result(vis_fol.path(), color_map)

    sh.eval()
    sh.show()
    sh.write_report(os.path.join(mf, "report.txt"))


def run_multi_training(df, mf, model, color_coding):

    if not os.path.isdir(mf):
        os.mkdir(mf)

    number_of_images = [4, 8, 16, 32, 64]  # , 4, 8, 16, 32, 64, 128
    iterations = 10

    for n in number_of_images:
        for i in range(iterations):
            sub_mf = os.path.join(mf, "-{}-RUN-{}".format(n, i))
            if not os.path.isdir(os.path.join(mf, "-{}-RUN-{}".format(n, i))):
                run_training(os.path.join(df, "train"), sub_mf, n, model, color_coding)
            if not os.path.isfile(os.path.join(sub_mf, "report.txt")):
                run_test(os.path.join(df, "test"), sub_mf, False)



def main():

    base_dir = "/Users/fmuenke/datasets/diss"
    base_model_dir = "/Users/fmuenke/ai_models"

    cc_crk = {"crack": [[0, 255, 0], [255, 0, 0]]}
    cc_pot = {"pothole": [[255, 0, 0], [255, 0, 0]]}
    cc_rtk = {
        "1": [[255, 85, 0], [255, 85, 0]], 
        "2": [[255, 170, 127], [255, 170, 127]], 
        "3": [[255, 255, 127], [255, 255, 127]], 
        "4": [[85, 85, 255], [85, 85, 255]], 
        "5": [[255, 255, 255], [255, 255, 255]], 
        "6": [[170, 0, 127], [170, 0, 127]], 
        "7": [[85, 170, 127], [85, 170, 127]], 
        "8": [[255, 85, 255], [255, 85, 255]], 
        "9": [[255, 0, 0], [255, 0, 0]], 
        "10": [[0, 0, 127], [0, 0, 127]], 
        "11": [[170, 0, 0], [170, 0, 0]]
    }

    cc_floot = {
        "Building-flooded": [[1, 1, 1], [255, 0, 0]],
        "Building-non-flooded": [[2, 2, 2], [0, 255, 0]],
        "Road-flooded": [[3, 3, 3], [0, 0, 255]],
        "Road-non-flooded": [[4, 4, 4], [255, 255, 0]], 
        "Water": [[5, 5, 5], [255, 0, 255]], 
        "Tree": [[6, 6, 6], [0, 255, 255]], 
        "Vehicle": [[7, 7, 7], [100, 100, 100]], 
        "Pool": [[8, 8, 8], [255, 255, 255]], 
        "Grass": [[9, 9, 9], [0, 100, 0]]
    }

    list_of_datasets = [
        # [os.path.join(base_dir, "crack500"), cc_crk],
        # [os.path.join(base_dir, "gaps384"), cc_crk],
        # [os.path.join(base_dir, "pothole600"), cc_pot],
        # [os.path.join(base_dir, "pothole-mix"), cc_crk],
        # [os.path.join(base_dir, "pothole-mix"), cc_pot],
        # [os.path.join(base_dir, "edmcrack600"), cc_crk],
        # [[os.path.join(base_dir, "CPRID"), cc_crk],
        # [[os.path.join(base_dir, "CPRID"), cc_pot],
        # [[os.path.join(base_dir, "CNR"), cc_pot],
        # [os.path.join(base_dir, "crack_segmentation"), cc_crk],
        # [os.path.join(base_dir, "crack_segmentation_vialytics"), cc_crk],
        # [os.path.join(base_dir, "RTK-street_surface_segmentation"), cc_rtk],
        [os.path.join(base_dir, "floodnet"), cc_floot],
    ]

    

    # DEFINE MODEL ###############
    model = model_cipp()
    ##############################

    for df, color_coding in list_of_datasets:
        dataset_id = os.path.basename(df)
        for cls in color_coding:
            single_cc = {cls: color_coding[cls]}
            mf = os.path.join(base_model_dir, f"cipp-{cls}-{dataset_id}")
            run_multi_training(df, mf, model, single_cc)



if __name__ == "__main__":
    main()
