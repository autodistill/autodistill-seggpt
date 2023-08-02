<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill SegGPT Module

This repository contains the code supporting the SegGPT base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[SegGPT](https://github.com/baaivision/Painter/tree/main/SegGPT) is a transformer-based few-shot semantic segmentation model developed by [BAAI Vision](https://github.com/baaivision).

This model performs well on task-specific segmentation tasks when given a few labeled images from which to learn features about the objects you want to identify.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SegGPT Autodistill documentation](https://autodistill.github.io/autodistill/base_models/seggpt/).

## Installation

To use SegGPT with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-seggpt
```

## About SegGPT

This Autodistill module uses a handful of pre-labelled images for improved accuracy.

You will need some labeled images to use SegGPT. Don't have any labeled images? Check out [Roboflow Annotate](https://roboflow.com/annotate), a feature-rich annotation tool from which you can export data for use with Autodistill.

## Quickstart

```python
from autodistill_seggpt import SegGPT,FewShotOntology

base_model = SegGPT(
    ontology=FewShotOntology(supervision_dataset)
)

base_model.label("./unlabelled-climbing-photos", extension=".jpg")
```

## How to load data from Roboflow

Labelling and importing images is easy!

You can use [Roboflow Annotate](https://roboflow.com/annotate) to label a few images (5-10 should work fine). For your Project Type, make sure to pick Instance Segmentation--you'll be labelling with polygons.

Once you've labelled your images, you can press Generate > Generate New Version. You can use all the default options--no Augmentations are necessary.

Once your dataset version is generated, you can press Export > Continue.

Then you'll get some download code to copy--it should look something like this:

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="ABCDEFG")
project = rf.workspace("lorem-ipsum").project("dolor-sit-amet")
dataset = project.version(1).download("yolov8")
```

To import your dataset into Autodistill, run the following:

```py
import supervision as sv
supervision_dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=f"{dataset.location}/train/images",
    annotations_directory_path=f"{dataset.location}/train/labels",
    data_yaml_path=f"{dataset.location}/data.yaml",
    force_masks=True
)
```

## License

The code in this repository is licensed under an [MIT license](LICENSE).

See the SegGPT repository for more information on the [SegGPT license](https://github.com/baaivision/Painter/tree/main).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
