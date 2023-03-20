import numpy as np
import cv2
from skimage.filters import frangi
from skimage.feature import canny

from conventional_image_processing_pipeline.layer_operations import normalize


class FillContours:
    list_of_parameters = [None, 1]
    key = "fill_contours"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img[x_img < 0.5] = 0
        x_img[x_img >= 0.5] = 1
        x_img = 255 * x_img
        cnt, _ = cv2.findContours(x_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x_img = cv2.fillPoly(x_img, pts=cnt, color=255)
        return x_img.astype(np.float64) / 255


class CannyEdgeDetector:
    list_of_parameters = [None, 3, 5, 9, 17]
    key = "canny_edge"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = canny(x_img.astype(np.uint8), sigma=self.parameter)
        return x_img.astype(np.float64) / 255


class FrangiFilter:
    list_of_parameters = [None, 1]
    key = "frangi"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = frangi(x_img)
        x_img = normalize(x_img)
        return x_img


class EdgeDetector:
    list_of_parameters = [None, 2+1, 4+1, 8+1, 16+1]
    key = "edge"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        x_img = cv2.Laplacian(np.copy(x_img.astype(np.uint8)), -1, ksize=self.parameter)
        x_img = normalize(x_img.astype(np.float64))
        return x_img


class EdgeSobel:
    list_of_parameters = [None, 2+1, 4+1, 8+1, 16+1]
    key = "edge_sobel"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S

        grad_x = cv2.Sobel(x_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(x_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        grad = normalize(grad.astype(np.float64))
        return grad
