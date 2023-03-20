import cv2
import numpy as np
from skimage.feature import blob_log
from skimage.segmentation import watershed


def mark_blobs(x_img, blobs):
    mask = np.zeros((x_img.shape[0], x_img.shape[1], 3))
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    for y, x, r in blobs:
        x = int(np.round(x))
        y = int(np.round(y))
        r = int(np.round(r))
        mask = cv2.circle(mask, (x, y), r, (1, 1, 1), -1)
    mask = np.max(mask, axis=2)
    return mask


def mark_blobs_with_watershed(x_img, blobs):
    blobs = np.round(blobs).astype(np.int)
    if len(blobs) == 0:
        return np.zeros(x_img.shape)
    x_img = 255 * x_img

    # cv2.imwrite("/Users/fmuenke/Desktop/test_1.png", x_img)
    markers = np.zeros((x_img.shape[0], x_img.shape[1], 3))
    for y, x, r in blobs:
        x, y, r = int(np.round(x)), int(np.round(y)), int(np.round(r))
        markers = cv2.circle(markers, (x, y), int(r * 2), (1, 1, 1), 1)

    for y, x, r in blobs:
        x, y, r = int(np.round(x)), int(np.round(y)), int(np.round(r))
        markers = cv2.circle(markers, (x, y), int(r * np.sqrt(2) / 3), (2, 2, 2), -1)
        # markers[y, x] = 2
    # cv2.imwrite("/Users/fmuenke/Desktop/test_2.png", markers * 128)
    markers = np.max(markers, axis=2)
    if np.min(markers) == 1:
        return markers - 1

    labels = watershed(x_img, markers)
    # cv2.imwrite("/Users/fmuenke/Desktop/test_3.png", labels * 128)
    return labels.astype(np.float64) - 1


class DetectBlob:
    list_of_parameters = [None, 1, 3, 7, 13, 17]
    key = "detect_blob"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        blobs = blob_log(x_img, min_sigma=self.parameter * 0.5, max_sigma=self.parameter, num_sigma=10, threshold=0.10)
        blob_mask = mark_blobs_with_watershed(x_img, blobs)
        return blob_mask
