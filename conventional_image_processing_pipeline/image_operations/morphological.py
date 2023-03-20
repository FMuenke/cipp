import numpy as np
import cv2


class MorphologicalOpening:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "opening"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return x_img.astype(np.float64) / 255


class NegativeMorphologicalOpening:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "negative_opening"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalOpening(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)


class MorphologicalErosion:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "erode"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_ERODE, kernel)
        return x_img.astype(np.float64) / 255


class NegativeMorphologicalErosion:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "negative_erode"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalErosion(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)


class MorphologicalDilatation:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "dilate"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_DILATE, kernel)
        return x_img.astype(np.float64) / 255


class NegativeMorphologicalDilatation:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "negative_dilate"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalDilatation(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)


class MorphologicalClosing:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "closing"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        kernel = np.ones((self.parameter, self.parameter), np.uint8)
        x_img = 255 * x_img
        x_img = cv2.morphologyEx(x_img.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return x_img.astype(np.float64) / 255


class NegativeMorphologicalClosing:
    list_of_parameters = [None, 2 + 1, 4 + 1, 8 + 1, 16 + 1]
    key = "negative_closing"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        op = MorphologicalClosing(self.parameter)
        x_img_morph = op.inference(x_img)
        return np.abs(x_img - x_img_morph)