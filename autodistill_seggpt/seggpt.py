model_url = "https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth"
ckpt_name = "seggpt_vit_large.pth"
ckpt_path = None
model = "seggpt_vit_large_patch16_input896x448"

import os
import subprocess
import sys
def check_dependencies():
    # Create the ~/.cache/autodistill directory if it doesn't exist
    autodistill_dir = os.path.expanduser("~/.cache/autodistill")
    os.makedirs(autodistill_dir, exist_ok=True)
    
    os.chdir(autodistill_dir)

    try:
        import detectron2
    except ImportError:
        print("Installing detectron2...")
        subprocess.run([sys.executable,"-m","pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])
        
    
    # Check if SegGPT is installed
    seggpt_path = os.path.join(autodistill_dir, "Painter","SegGPT","SegGPT_inference")
    models_dir = os.path.join(seggpt_path, "models")
    global ckpt_path
    ckpt_path = os.path.join(models_dir, ckpt_name)
    sys.path.append(seggpt_path)

    if not os.path.isdir(seggpt_path):
        print("Installing SegGPT...")
        subprocess.run(["git", "clone", "https://github.com/baaivision/Painter.git"])

        os.makedirs(models_dir, exist_ok=True)

        print("Downloading SegGPT weights...")
        subprocess.run(["wget", model_url, "-O", ckpt_path])


check_dependencies()

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import DetectionBaseModel
from PIL import Image

# SegGPT repo files
from seggpt_engine import run_one_image
from seggpt_inference import prepare_model

from segment_anything import SamPredictor
from supervision import Detections
from torch.nn import functional as F

# Model/dataset parameters - don't need to be configurable

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res, hres = 448, 448

# SegGPT-specific utils
from . import colors
from .few_shot_ontology import FewShotOntology
from .postprocessing import bitmasks_to_detections, quantize, quantized_to_bitmasks
from .sam_refine import refine_detections,load_SAM

use_colorings = colors.preset != "white"

@dataclass
class SegGPT(DetectionBaseModel):
    sam_predictor: Union[None, SamPredictor] = None
    model: Union[None, torch.nn.Module] = None

    def __init__(
        self,
        ontology: FewShotOntology,
        refine_detections: bool = True,
        sam_predictor=None,
    ):

        self.ontology = ontology
        self.refine_detections = refine_detections

        self.load_models(sam_predictor)

        self.ref_imgs = {}

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, dsize=(res, hres))
        img = img / 255.0
        img = img - imagenet_mean
        img = img / imagenet_std
        return img

    # convert an img + detections into an img + mask.
    # note: all the detections have the same class in the FewShotOntology.
    def prepare_ref_img(self, img: np.ndarray, detections: Detections):
        ih, iw, _ = img.shape
        og_img = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocess(img)
        img = self.imagenet_preprocess(img)

        # draw masks onto image
        mask = np.zeros_like(og_img)
        for detection in detections:
            curr_rgb = colors.next_color()
            det_box, det_mask, *_ = detection

            mh, mw = det_mask.shape

            if not (mh == ih and mw == iw):
                # mask is not the same size as the image. resize it.
                raise NotImplementedError(
                    "mask resizing is disabled--seems to cause kernel crashes."
                )
                det_mask = cv2.resize(det_mask, dsize=(iw, ih))
                mh, mw = det_mask.shape
            assert (
                mh == ih and mw == iw
            ), f"mask shape {det_mask.shape} does not match image shape {og_img.shape}. new image shape is {img.shape}"
            assert det_mask.max() <= 1, "mask values should be in [0,1]"

            mask[det_mask] = curr_rgb

        mask = self.preprocess(mask)

        return img, mask

    # Convert a list of reference images into a SegGPT-friendly batch.
    def prepare_ref_imgs(
        self, cls_name: str, refs: List[Tuple[np.ndarray, Detections]]
    ):
        if cls_name in self.ref_imgs:
            return self.ref_imgs[cls_name]

        imgs, masks = [], []
        min_area = inf
        for ref_img, detections in refs:
            img, mask = self.prepare_ref_img(ref_img, detections)

            img_min_area = detections.area.min()
            min_area = min(min_area, img_min_area)

            imgs.append(img)
            masks.append(mask)
        imgs = np.stack(imgs, axis=0)
        masks = np.stack(masks, axis=0)
        ret = (imgs, masks, min_area)
        self.ref_imgs[cls_name] = ret

        return ret

    @torch.no_grad()
    def predict(
        self, input: Union[str, np.ndarray], _confidence: int = 0.5
    ) -> sv.Detections:
        if type(input) == str:
            if input in self.ontology.ref_dataset.images:
                image = Image.fromarray(self.ref_dataset.images[input])
            image = Image.open(input).convert("RGB")
        else:
            image = Image.fromarray(input)

        size = image.size
        input_image = np.array(image)

        img = self.preprocess(input_image)

        detections = []
        for keyId, raw_ref_imgs in enumerate(self.ontology.rich_prompts()):
            assert len(img.shape) == 3, f"img.shape: {img.shape}"

            ref_imgs, ref_masks, min_ref_area = self.prepare_ref_imgs(
                keyId, raw_ref_imgs
            )

            # convert ref_imgs from (N,H,W,C) to (N,2H,W,C)
            img_repeated = np.repeat(img[np.newaxis, ...], len(ref_imgs), axis=0)

            # SegGPT uses this weird format--it needs images/masks to be in format (N,2H,W,C)--where the first H rows are the reference image, and the next H rows are the input image.
            imgs = np.concatenate((ref_imgs, img_repeated), axis=1)
            masks = np.concatenate((ref_masks, ref_masks), axis=1)

            torch.manual_seed(2)
            output = run_one_image(imgs, masks, self.model, device)
            output = (
                F.interpolate(
                    output[None, ...].permute(0, 3, 1, 2),
                    size=[size[1], size[0]],
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)[0]
                .numpy()
            )

            # We constrain all masks to follow a given color palette.
            # This can help distinguish adjacent instances.
            # But it also serves as a bitmask-ifier when we just set the palette to be white.
            quant_output = quantize(output)

            to_bitmask_output = quant_output
            to_bitmask_palette = colors.palette

            bitmasks = quantized_to_bitmasks(to_bitmask_output, to_bitmask_palette)
            new_detections = bitmasks_to_detections(bitmasks, keyId)

            new_detections = new_detections[new_detections.area > min_ref_area * 0.75]

            if len(new_detections) > 0:
                detections.append(new_detections)

        # filter <100px detections
        detections = Detections.merge(detections)

        if len(detections) > 0:
            detections = detections[has_polygons(detections.mask)]
            if self.refine_detections:
                detections = refine_detections(
                    input_image, detections, self.sam_predictor
                )

        return detections
    
    # Load SegGPT and SAM.
    # We share these models globally across all SegGPT instances, since we end up making lots of SegGPT instances during find_best_examples.
    def load_models(self,sam_predictor):
        if SegGPT.model is None:
            SegGPT.model = prepare_model(ckpt_path, model, colors.seg_type).to(device)
        self.model = SegGPT.model

        # We load the SAM predictor if it's a) needed and b) not already loaded.
        if sam_predictor is not None or not self.refine_detections:
            self.sam_predictor = sam_predictor
        else:
            if SegGPT.sam_predictor is None:
                SegGPT.sam_predictor = load_SAM()
            self.sam_predictor = SegGPT.sam_predictor



from supervision.dataset.utils import approximate_mask_with_polygons


def has_polygons(masks: np.ndarray) -> np.ndarray:
    n, h, w = masks.shape

    ret = np.zeros((n,), dtype=bool)
    for i in range(n):
        mask = masks[i]
        polygons = approximate_mask_with_polygons(mask)
        if len(polygons) > 0:
            ret[i] = True

    return ret
