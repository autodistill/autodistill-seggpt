from dataclasses import dataclass
from math import inf
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.core import Ontology
from autodistill.detection import DetectionBaseModel, DetectionOntology
from PIL import Image

# SegGPT repo files
from seggpt.seggpt_engine import run_one_image
from seggpt.seggpt_inference import prepare_model

# SAM files
from segment_anything import SamPredictor
from supervision import Detections
from supervision.dataset.core import DetectionDataset
from torch.nn import functional as F

# Model/dataset parameters - don't need to be configurable

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

ckpt_path = "seggpt_vit_large.pth"
device = torch.device("cuda")
model = "seggpt_vit_large_patch16_input896x448"

res, hres = 448, 448

# SegGPT-specific utils
from . import colors
from .few_shot_ontology import FewShotOntology
from .postprocessing import bitmasks_to_detections, quantize, quantized_to_bitmasks
from .sam_refine import load_sam, refine_detections

use_colorings = True


def show_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)

    print(
        f"Total: {t//(10**9)}, Reserved: {r//(10**9)}, Allocated: {a//(10**9)}, Free: {(t-a)//(10**9)}"
    )


class SegGPT(DetectionBaseModel):
    sam_predictor: Union[None, SamPredictor] = None
    model: Union[None, torch.nn.Module] = None

    def __init__(
        self,
        ontology: FewShotOntology,
        refine_detections: bool = True,
        sam_predictor=None,
    ):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ontology = ontology
        self.refine_detections = refine_detections

        if SegGPT.model is None:
            SegGPT.model = prepare_model(ckpt_path, model, colors.seg_type).to(
                self.DEVICE
            )
        self.model = SegGPT.model

        if sam_predictor is not None:
            self.sam_predictor = sam_predictor
        else:
            if SegGPT.sam_predictor is None:
                SegGPT.sam_predictor = load_sam()
            self.sam_predictor = SegGPT.sam_predictor

        self.ref_imgs = {}

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, dsize=(res, hres))
        img = img / 255.0
        return img

    def imagenet_preprocess(self, img: np.ndarray) -> np.ndarray:
        img = img - imagenet_mean
        img = img / imagenet_std
        return img

    # convert an img + detections into an img + mask.
    # note: all the detections map to the same base-model class in the Ontology--they can be treated as the same class.
    def prepare_ref_img(self, img: np.ndarray, detections: Detections):
        ih, iw, _ = img.shape
        og_img = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preprocess(img)
        img = self.imagenet_preprocess(img)

        # draw masks onto image
        mask = np.zeros_like(og_img)
        colors.reset_colors()
        for detection in detections:
            curr_rgb = colors.next_rgb()
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
        mask = self.imagenet_preprocess(mask)

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
        img = self.imagenet_preprocess(img)  # shape (H,W,C)

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
            # show_usage()

            for i in range(len(imgs)):
                cv2.imwrite(
                    f"debug/img_{i}.png", (imgs[i] * imagenet_std + imagenet_mean) * 255
                )
                cv2.imwrite(
                    f"debug/mask_{i}.png",
                    (masks[i] * imagenet_std + imagenet_mean) * 255,
                )

            torch.manual_seed(2)
            output = run_one_image(imgs, masks, self.model, self.DEVICE)
            output = (
                F.interpolate(
                    output[None, ...].permute(0, 3, 1, 2),
                    size=[size[1], size[0]],
                    mode="nearest",
                )
                .permute(0, 2, 3, 1)[0]
                .numpy()
            )

            quant_output = quantize(output)

            if use_colorings:
                to_bitmask_output = quant_output
                to_bitmask_palette = colors.palette
            else:
                to_bitmask_output = quant_output.sum(axis=-1) > 10
                to_bitmask_output = to_bitmask_output[..., None] * 255
                to_bitmask_palette = np.asarray([[255, 255, 255]])

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
