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

It thrives when given multiple labelled example images.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SegGPT Autodistill documentation](https://autodistill.github.io/autodistill/base_models/seggpt/).

## Installation

To use SegGPT with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-seggpt
```

## About SegGPT

This Autodistill module uses a handful of pre-labelled images for improved accuracy.

Note: this adds some complexity--`autodistill_seggpt` is BYOD--bring your own dataset.

## Quickstart

```python
from autodistill_seggpt import SegGPT,FewShotOntology
import supervision as sv

#
# download dataset with a few labelled images (5-10 is recommended, but you can go as high as you like)
#
from roboflow import login,Roboflow

login()
rf = Roboflow()

project = rf.workspace("roboflow-4rfmv").project("climbing-y56wy")
dataset = project.version(5).download("coco-segmentation")

climbing_dataset = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
    force_masks=True
)

#
# Create ontology
#

climbing_ontology = FewShotOntology(climbing_dataset)

base_model = SegGPT(
    ontology=climbing_ontology,
    refine_detections=True
)

base_model.label("./unlabelled-climbing-photos", extension=".jpg")
```

## License

The code in this repository is licensed under an [MIT license](LICENSE).

See the SegGPT repository for more information on the [SegGPT license](https://github.com/baaivision/Painter/tree/main).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!