from supervision import DetectionDataset,Detections
import numpy as np

from typing import Dict,Tuple,Callable

class MetricDirection:
    LOWER_IS_BETTER = 0
    HIGHER_IS_BETTER = 1

Metric = Dict[str,Tuple[Callable,str,int]]

metrics_registry:Metric = {}

eps = 1e-6

def validate_datasets(gt_dataset:DetectionDataset,pred_dataset:DetectionDataset):
    assert gt_dataset.classes == pred_dataset.classes, f"gt classes: {gt_dataset.classes}, pred classes: {pred_dataset.classes}"
    assert gt_dataset.images.keys() == pred_dataset.images.keys()

def iou(gt_dataset:DetectionDataset,pred_dataset:DetectionDataset)->float:
    validate_datasets(gt_dataset,pred_dataset)

    running_intersection = 0
    running_union = 0

    for img_name in gt_dataset.images:
        gt_detections = gt_dataset.annotations[img_name]
        pred_detections = pred_dataset.annotations[img_name]

        img = gt_dataset.images[img_name]

        gt_mask = get_combined_mask(img,gt_detections)
        pred_mask = get_combined_mask(img,pred_detections)

        intersection = np.sum(gt_mask*pred_mask)
        union = np.sum(np.logical_or(gt_mask,pred_mask))

        running_intersection += intersection
        running_union += union
    
    return running_intersection/(running_union+eps)

metrics_registry["iou"] = (iou,"IoU",MetricDirection.HIGHER_IS_BETTER)

def get_combined_mask(img:np.ndarray,detections:Detections)->np.ndarray:
    mask = np.zeros(img.shape[:2],dtype=np.uint8)

    for detection in detections:
        det_box,det_mask,*_ = detection
        mask[det_mask.astype(bool)] = 1

    mask = np.clip(mask,0,1)

    return mask

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys, os

old_stdout = sys.stdout

# Disable
def blockPrint():
    return
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = old_stdout


last_coco_eval = None

# For now, save both as COCO format, then use pycocotoosl COCOeval
def mAP(gt_dataset:DetectionDataset,pred_dataset:DetectionDataset)->float:
    validate_datasets(gt_dataset,pred_dataset)

    gt_filename = "gt.json"
    pred_filename = "pred.json"

    # don't save images for either--mAP only depends on the masks
    gt_dataset.as_coco(annotations_path=gt_filename)
    pred_dataset.as_coco(annotations_path=pred_filename)

    blockPrint()

    gt_coco = COCO(gt_filename)
    pred_coco = COCO(pred_filename)

    for ann in pred_coco.anns.values():
        ann["score"] = 1

    coco_eval = COCOeval(gt_coco,pred_coco,"segm")
    coco_eval.evaluate()
    coco_eval.accumulate()

    coco_eval.summarize()

    enablePrint()

    final_mAP = coco_eval.stats[0]

    global last_coco_eval
    last_coco_eval = coco_eval

    return final_mAP

metrics_registry["map"] = (mAP,"mAP",MetricDirection.HIGHER_IS_BETTER)