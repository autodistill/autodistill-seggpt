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

```

```python
from autodistill_detic import DETIC

# define an ontology to map class names to our DETIC prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = DETIC(
    ontology=CaptionOntology(
        {
            "person": "person",
        }
    )
)
base_model.label("./context_images", extension=".jpg")
```

## License

The code in this repository is licensed under an [MIT license](LICENSE).

See the Meta Research DETIC repository for more information on the [DETIC license](https://github.com/facebookresearch/Detic).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!