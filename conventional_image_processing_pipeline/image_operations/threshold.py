import numpy as np
from skimage.filters import threshold_otsu, threshold_local


class Threshold:
    list_of_parameters = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    key = "threshold"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img[x_img < self.parameter] = 0
        x_img[x_img >= self.parameter] = 1
        return x_img


class ThresholdPercentile:
    list_of_parameters = [70, 80, 90, 95, 99]
    key = "threshold_percentile"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        threshold = np.percentile(x_img, self.parameter)
        x_img[x_img < threshold] = 0
        x_img[x_img >= threshold] = 1
        return x_img


class ThresholdOtsu:
    list_of_parameters = [None, 1]
    key = "threshold_otsu"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        if len(np.unique(x_img)) == 1:
            return np.zeros(x_img.shape)
        threshold = threshold_otsu(x_img)
        x_img[x_img < threshold] = 0
        x_img[x_img >= threshold] = 1
        return x_img


class LocalThreshold:
    list_of_parameters = [None, 8+1, 16+1, 32+1, 64+1]
    key = "local_threshold"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        threshold = threshold_local(x_img, block_size=self.parameter)
        x_img[x_img < threshold] = 0
        x_img[x_img >= threshold] = 1
        return x_img
