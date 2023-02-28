# Conventional Image Processing Pipelines

This repository provides a simple and efficient means of semantic image segmentation.
For this purpose we define a sequence of conventional image processing operations (like thresholding, edge-detection, watershed, ...).

Each operation contains a predefined set of parameters which are adapted for the optimal output based on the training images.

## Setup

It is recommended to use miniconda for setting up the repository.

```bash
conda create -n synth python=3.9
conda activate synth
pip install -r requirements.txt
```


## Usage

The Repository provides all resources to apply a conventional image processing pipeline (CIPP) to a given semantic segmentation problem.

A CIPP-model is defined by defining a graph.
````Python
from conventional_image_processing_pipeline import Model
from conventional_image_processing_pipeline import InputLayer
from conventional_image_processing_pipeline import CIPPLayer

x_input = InputLayer("INPUT-LAYER-NAME", features_to_use=["RGB-color"])
x_cipp = CIPPLayer(
    x_input, "CIPP-LAYER", 
    operations=[
        "OPERATION_STEP_1",
        "OPERATION_STEP_2",
        ["OPERATION_STEP_3.1", "OPERATION_STEP_3.2"],
    ], 
    selected_layer=[0, 1, 2],
    optimizer="OPTIMIZER-TYPE", 
    use_multiprocessing=True
)
cipp_model = Model(graph=x_cipp)
````

### InputLayer
The input layer is the entry point of the graph and defines the initial features to use and the size of the input image.
In this example we are forwarding the image as a regular RGB-Image with all three channels. Other options are: (gray-color, HSV-color and opponent-color).
The image size can be adjusted in two ways:
- initial_downscale: (int) The amount of time the image size is halved. Example: 1 --> image height and width are halved.
- height and width: Set a fixed image height and or width. If only one is set the aspect ratio is kept.

### CIPPLayer
The cipp layer contains the separate image processing operations. In this layer we define the operations to use for the problem at hand as well as the order.
Each operation can be deactivated during training. It is as well possible to have the model choose in one step from a group of operations by providing a list of operations for a step instead of just one operation.
The operations are optimized by a set optimizer. (Options: grid_search, random_search, genetic_algorithm).

The input layer returns based on the features-option a different amount of channels. With the "selected_layer" flag it is possible to select all or just a subset of channels to consider during training.

### Model
The model is used as a wrapper to train and run the pipeline.


## Data
The data should be organized as follows:
- Folder "images" containing all separate images (img1.png, img2.png)
- Folder "labels" containing all corresponding labels parallel to it (img1.png, img2.png)
The filename (excluding the suffix) should be the same for image and corresponding label.

The label-format is defined by the variable color_coding where the first triplet corresponds to the values of that class in the label map
and the second triplet corresponds to the final prediction coloring (only used for visualization).

In the following we showcase an example to load data and use it for training.

````python
from data_structure import SegmentationDataSet

color_coding = {"class": [[255, 255, 255], [255, 0, 0]]}

d_set = SegmentationDataSet("PATH-TO-DATA", color_coding)
tag_set = d_set.load()
train_set, validation_set = d_set.split(tag_set, percentage=0.1, random=True)

model.fit(train_set, validation_set)
model.save("PATH-TO-SAVE-THE-MODEL")
````