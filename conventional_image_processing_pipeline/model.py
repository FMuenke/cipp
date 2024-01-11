import os
from time import time
from tqdm import tqdm
import numpy as np
import cv2
from utils.utils import check_n_make_dir, load_dict

from conventional_image_processing_pipeline.input_layer import InputLayer
from conventional_image_processing_pipeline.cipp_layer import CIPPLayer

from data_structure.model_statistics import ModelStatistics
from utils.segmention_mask import convert_cls_to_color, side_by_side


class Model:
    def __init__(self, graph):
        self.graph = graph

        self.description = dict()

    def fit(self, train_tags, validation_tags):
        print("===============================")
        print("=====Begin Model Training======")
        print("===============================")
        self.graph.fit(train_tags, validation_tags)

    def save(self, model_path):
        check_n_make_dir(model_path)
        check_n_make_dir(os.path.join(model_path, "graph"))
        self.graph.save(os.path.join(model_path, "graph"))
        print("Model was saved to: {}".format(model_path))

    def load(self, model_path):
        print("Loading Model from: {}".format(model_path))
        if os.path.isdir(model_path):
            if os.path.isdir(os.path.join(model_path, "graph")):
                graph_start = os.listdir(os.path.join(model_path, "graph"))[0]
                layer = self.load_layer(os.path.join(model_path, "graph", graph_start))
                layer.load(os.path.join(model_path, "graph", graph_start))
                self.graph = layer

        print("Model was loaded:")
        print(self.graph)

    def predict(self, data):
        return self.graph.predict(data)

    def explain(self, data):
        return self.graph.explain(data)

    def evaluate(self, tags, color_coding, results_folder):
        print("[INFO] Begin Model Evaluation")
        check_n_make_dir(results_folder)
        res_folder = os.path.join(results_folder, "segmentations")
        check_n_make_dir(res_folder, clean=True)
        vis_folder = os.path.join(results_folder, "overlays")
        check_n_make_dir(vis_folder, clean=True)
        exp_folder = os.path.join(results_folder, "explain")
        check_n_make_dir(exp_folder, clean=True)
        sbs_folder = os.path.join(results_folder, "side_by_side")
        check_n_make_dir(sbs_folder, clean=True)

        t0 = time()
        sh = ModelStatistics(color_coding)
        for tag in tqdm(tags):
            image = tag.load_x()
            cls_map = self.predict(image)
            exp_img = self.explain(image)
            cv2.imwrite(
                os.path.join(exp_folder, "{}.png".format(tag.id)),
                exp_img
            )
            color_map = convert_cls_to_color(cls_map, color_coding, unsupervised=False)
            tag.write_result(res_folder, color_map)
            tag.eval(color_map, sh)
            tag.visualize_result(vis_folder, color_map)
            cv2.imwrite(
                os.path.join(sbs_folder, "{}.png".format(tag.id)),
                side_by_side(image, color_map)
            )

        with open(os.path.join(results_folder, "time_prediction.txt"), "w") as f:
            f.write("[INFO] done in %0.3fs" % (time() - t0))
        sh.eval()
        sh.show()
        sh.write_report(os.path.join(results_folder, "report.txt"))

    def load_layer(self, model_folder):
        opt = load_dict(os.path.join(model_folder, "opt.json"))
        if "layer_type" not in opt:
            raise ValueError("No LayerType Option is defined!")

        if opt["layer_type"] == "INPUT_LAYER":
            layer = InputLayer(opt["name"], opt["features_to_use"], height=opt["height"], width=opt["width"],
                               initial_down_scale=opt["down_scale"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        if opt["layer_type"] == "CIPP_LAYER":
            prev_layer = self.load_previous_layers(model_folder)
            if "selected_layer" not in opt:
                opt["selected_layer"] = -1
            layer = CIPPLayer(prev_layer, opt["name"], None, selected_layer=opt["selected_layer"])
            layer.set_index(int(opt["index"]))
            layer.load(model_folder)
            return layer

        raise ValueError("Layer: {} not recognised!".format(opt["layer_type"]))

    def load_previous_layers(self, model_folder):
        p_layer = []
        for path in os.listdir(model_folder):
            prev_path = os.path.join(model_folder, path)
            if os.path.isdir(prev_path):
                layer = self.load_layer(prev_path)
                p_layer.append(layer)

        p_layer_sorted = [0] * len(p_layer)
        for layer in p_layer:
            p_layer_sorted[int(layer.index)] = layer
        return p_layer


