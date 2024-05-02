from conventional_image_processing_pipeline.image_operations import *


class Watershed:
    list_of_parameters = [None, 1, 5]
    key = "watershed"

    def __init__(self, parameter):
        self.parameter = parameter

    def inference(self, x_img):
        if self.parameter is None:
            return x_img
        x_img = 255 * x_img
        markers = np.zeros(x_img.shape, dtype=np.uint)
        markers[x_img <= np.percentile(x_img, self.parameter)] = 1
        markers[x_img >= np.percentile(x_img, 100 - self.parameter)] = 2
        if np.min(markers) == 1:
            return markers - 1

        labels = watershed(x_img, markers)
        return labels.astype(np.float64) - 1


LIST_OF_OPERATIONS = [
    DetectBlob,
    Invert,
    Cropping,
    EdgeDetector,
    FrangiFilter,
    Threshold,
    MorphologicalOpening,
    NegativeMorphologicalOpening,
    MorphologicalClosing,
    NegativeMorphologicalClosing,
    ThresholdPercentile,
    MorphologicalDilatation,
    NegativeMorphologicalDilatation,
    MorphologicalErosion,
    NegativeMorphologicalErosion,
    Blurring,
    TopClipping,
    TopClippingPercentile,
    BottomClipping,
    BottomClippingPercentile,
    CannyEdgeDetector,
    LocalNormalization,
    RemoveSmallObjects,
    RemoveBigObjects,
    RemoveSmallHoles,
    ThresholdOtsu,
    LocalThreshold,
    FillContours,
    Watershed,
    SelectSpheres,
    SelectSolid,
    EdgeSobel,
]
