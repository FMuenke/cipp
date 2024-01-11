import cv2
import numpy as np

from conventional_image_processing_pipeline.layer_operations import resize
from skimage.morphology import remove_small_objects, remove_small_holes


class LocalNormalization:
    list_of_parameters = [None, 4+1, 8+1, 16+1, 32+1]
    key = "local_normalization"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        total_avg = np.mean(x_img)
        avg = cv2.filter2D(
            np.copy(x_img), -1,
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.parameter, self.parameter))
        )
        img_norm = x_img - total_avg + avg
        return img_norm.astype(np.float64) / 255


class RemoveSmallObjects:
    list_of_parameters = [None, 8, 32, 128, 256, 512]
    key = "remove_small_objects"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = x_img.astype(np.int)
        if len(np.unique(x_img)) == 1:
            return x_img
        x_img = remove_small_objects(x_img.astype(np.bool), min_size=self.parameter)
        return x_img.astype(np.float64)


class RemoveBigObjects:
    list_of_parameters = [None, 64, 128, 256, 512]
    key = "remove_big_objects"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = x_img.astype(np.int)
        if len(np.unique(x_img)) == 1:
            return x_img
        x_img_stamp = np.copy(x_img)
        x_img = remove_small_objects(x_img.astype(np.bool), min_size=self.parameter)
        x_img_stamp = x_img_stamp - x_img
        return x_img_stamp.astype(np.float64)


class RemoveSmallHoles:
    list_of_parameters = [None, 8, 32, 128, 256, 512]
    key = "remove_small_holes"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = x_img.astype(np.int)
        if len(np.unique(x_img)) == 1:
            return x_img
        x_img = remove_small_holes(x_img.astype(np.bool), area_threshold=self.parameter)
        return x_img.astype(np.float64)


class Blurring:
    list_of_parameters = [None, 2+1, 4+1, 8+1, 16+1]
    key = "blurring"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = cv2.blur(np.copy(x_img.astype(np.uint8)), ksize=(self.parameter, self.parameter))
        return x_img.astype(np.float64) / 255


class Invert:
    list_of_parameters = [-1, 1]
    key = "invert"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter == -1:
            stamp = np.ones(x_img.shape) * np.max(x_img)
            return stamp - x_img
        else:
            return x_img


class Cropping:
    list_of_parameters = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    key = "crop"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter == 0.0:
            return x_img
        height, width = x_img.shape[:2]
        w_crp = int(self.parameter * width)
        h_crp = int(self.parameter * height)
        crp_area = np.ones((height, width))
        crp_area[h_crp:height-h_crp, w_crp:width-w_crp] = 0
        x_img[crp_area == 1] = 0
        # cv2.imwrite("./test/crp_{}.png".format(self.parameter), 255 * x_img)
        return x_img


class Resize:
    list_of_parameters = [None, 2, 4, 8]
    key = "resize"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        height, width = x_img.shape[:2]
        return resize(x_img, width=int(width / self.parameter), height=int(height / self.parameter))


class TopClipping:
    list_of_parameters = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    key = "top_clipping"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img[x_img > self.parameter] = 0
        return x_img


class TopClippingPercentile:
    list_of_parameters = [None, 1, 5, 10, 25, 50]
    key = "top_clipping_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = np.clip(x_img, 0, np.percentile(x_img, 100 - self.parameter))
        return x_img


class BottomClippingPercentile:
    list_of_parameters = [None, 1, 5, 10, 25, 50]
    key = "bottom_clipping_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = np.clip(x_img, np.percentile(x_img, self.parameter), 1)
        return x_img


class BottomClipping:
    list_of_parameters = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    key = "bottom_clipping"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img[x_img < self.parameter] = 0
        return x_img


class BottomClippingPercentile:
    list_of_parameters = [None, 1, 5, 10, 25, 50]
    key = "bottom_clipping_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = np.clip(x_img, np.percentile(x_img, self.parameter), 1)
        return x_img
