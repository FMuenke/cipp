import numpy as np
from skimage import measure

class SelectSpheres:
    list_of_parameters = [None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    key = "select_sphere"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        if len(np.unique(x_img)) == 1:
            return x_img
        mask = np.zeros(x_img.shape)
        x_img[x_img < 0.5] = 0
        x_img[x_img >= 0.5] = 1
        label_img = measure.label(x_img)
        regions = measure.regionprops(label_img)
        for props in regions:
            if props.eccentricity < self.parameter:
                mask[props.coords[:, 0], props.coords[:, 1]] = 1
        return mask


class SelectSolid:
    list_of_parameters = [None, 0.5, 0.6, 0.7, 0.8, 0.9]
    key = "select_solid"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        if len(np.unique(x_img)) == 1:
            return x_img
        mask = np.zeros(x_img.shape)
        x_img[x_img < 0.5] = 0
        x_img[x_img >= 0.5] = 1
        label_img = measure.label(x_img)
        regions = measure.regionprops(label_img)
        for props in regions:
            if props.solidity > self.parameter:  # solidity
                mask[props.coords[:, 0], props.coords[:, 1]] = 1
        return mask
