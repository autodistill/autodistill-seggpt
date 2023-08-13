from typing import Callable, Dict, Tuple

import numpy as np
from supervision import DetectionDataset, Detections


class MetricDirection:
    LOWER_IS_BETTER = 0
    HIGHER_IS_BETTER = 1


eps = 1e-6

class Metric:

    @staticmethod
    def evaluate_detections(self, gt_dets: Detections, pred_dets: Detections):
        raise NotImplementedError()
    @staticmethod
    def evaluate_datasets(self, gt_dataset: DetectionDataset, pred_dataset: DetectionDataset):
        raise NotImplementedError()
    @staticmethod
    def name()->str:
        raise NotImplementedError()
    @staticmethod
    def direction()->int:
        raise NotImplementedError()

metrics_registry: Dict[str,Metric] = {}
        
class DetectionMetric(Metric):
    @staticmethod
    def evaluate_datasets(self, gt_dataset: DetectionDataset, pred_dataset: DetectionDataset):
        DetectionMetric.validate_datasets(gt_dataset, pred_dataset)

        running_metric = 0
        running_count = 0

        for img_name in gt_dataset.images:
            gt_detections = gt_dataset.annotations[img_name]
            pred_detections = pred_dataset.annotations[img_name]

            metric = self.evaluate_detections(gt_detections, pred_detections)

            running_metric += metric
            running_count += 1
        
        return running_metric / (running_count + eps)

    @staticmethod
    def validate_datasets(gt_dataset: DetectionDataset, pred_dataset: DetectionDataset):
        assert (
            gt_dataset.classes == pred_dataset.classes
        ), f"gt classes: {gt_dataset.classes}, pred classes: {pred_dataset.classes}"
        assert gt_dataset.images.keys() == pred_dataset.images.keys()

class DatasetMetric(Metric):
    @staticmethod
    def evaluate_detections(self, gt_dets: Detections, pred_dets: Detections):

        if len(gt_dets) == 0:
            return 0

        max_class_id = np.concatenate([gt_dets.class_id, pred_dets.class_id]).max()

        h,w = gt_dets.mask.shape[1:]

        gt_dataset = DetectionDataset(
            classes = list(range(max_class_id+1)),
            images = {"test":np.zeros((h,w,3))},
            annotations={"test":gt_dets}
        )

        pred_dataset = DetectionDataset(
            classes = list(range(max_class_id+1)),
            images = {"test":np.zeros((h,w,3))},
            annotations={"test":pred_dets}
        )

        return self.evaluate_datasets(gt_dataset, pred_dataset)


#
#  IoU
#

class IoU(DatasetMetric):
    @staticmethod
    def name():
        return "IoU"
    @staticmethod
    def direction():
        return MetricDirection.HIGHER_IS_BETTER
    @staticmethod
    def evaluate_datasets(self, gt_dataset: DetectionDataset, pred_dataset: DetectionDataset):
        DetectionMetric.validate_datasets(gt_dataset, pred_dataset)

        running_intersection = 0
        running_union = 0

        for img_name in gt_dataset.images:
            gt_detections = gt_dataset.annotations[img_name]
            pred_detections = pred_dataset.annotations[img_name]

            if(len(gt_detections) == 0 and len(pred_detections) == 0):
                continue

            img = gt_dataset.images[img_name]

            gt_mask = get_combined_mask(img, gt_detections)
            pred_mask = get_combined_mask(img, pred_detections)

            intersection = np.sum(gt_mask * pred_mask)
            union = np.sum(np.logical_or(gt_mask, pred_mask))

            running_intersection += intersection
            running_union += union

        return running_intersection / (running_union + eps)

metrics_registry["iou"] = IoU


def get_combined_mask(img: np.ndarray, detections: Detections) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for detection in detections:
        det_box, det_mask, *_ = detection
        mask[det_mask.astype(bool)] = 1

    mask = np.clip(mask, 0, 1)

    return mask

#
# mAP
#

import os
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

oneify_score = False

class MaP(DatasetMetric):
    last_coco_eval = None
    old_stdout = None


    @staticmethod
    def name():
        return "mAP"

    @staticmethod
    def direction():
        return MetricDirection.HIGHER_IS_BETTER

    @staticmethod
    def evaluate_datasets(self, gt_dataset: DetectionDataset, pred_dataset: DetectionDataset):
        DatasetMetric.validate_datasets(gt_dataset, pred_dataset)

        gt_filename = "gt.json"
        pred_filename = "pred.json"

        # don't save images for either--mAP only depends on the masks
        gt_dataset.as_coco(annotations_path=gt_filename)
        pred_dataset.as_coco(annotations_path=pred_filename)

        MaP.blockPrint()

        gt_coco = COCO(gt_filename)
        pred_coco = COCO(pred_filename)

        if oneify_score:
            for ann in pred_coco.anns.values():
                ann["score"] = 1

        coco_eval = COCOeval(gt_coco, pred_coco, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()

        coco_eval.summarize()

        MaP.enablePrint()

        final_mAP = coco_eval.stats[0]

        MaP.last_coco_eval = coco_eval

        return final_mAP
    

    # Disable
    @classmethod
    def blockPrint(cls):
        cls.old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    # Restore
    @classmethod
    def enablePrint(cls):
        sys.stdout = cls.old_stdout

metrics_registry["map"] = MaP