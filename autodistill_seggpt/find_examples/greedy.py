import math
from random import sample
from typing import List, Type, Union, Callable

import numpy as np
import supervision as sv
from autodistill.detection import DetectionBaseModel
from supervision.dataset.core import DetectionDataset
from tqdm import tqdm

from .common import separable
from ..dataset_utils import (
    extract_images_from_dataset,
    extract_images_from_dataset,
    shrink_dataset_to_size,
    viz_dataset,
)
from ..few_shot_ontology import FewShotOntology
from ..metrics import Metric, MetricDirection

from random import choice
import json
import os


@separable
def grow_ontology(
    # general params
    ref_dataset: DetectionDataset,
    make_model: Callable[[FewShotOntology], DetectionBaseModel],
    metric: Metric,

    # sample-specific params
    num_examples: int = 4,
    num_trials: int = 2,
    max_test_imgs: int = 3,
)->FewShotOntology:

    positive_examples = [img_name for img_name in ref_dataset.images if len(ref_dataset.annotations[img_name]) > 0]

    valid_dataset = shrink_dataset_to_size(ref_dataset, max_test_imgs)
    num_examples = min(num_examples, len(ref_dataset.images))

    ontologies_scores = []

    for j in range(num_trials):
        curr_images = [choice(positive_examples)]

        avg_score = 0

        os.makedirs(f"greedies/{j}",exist_ok=True)

        for i in range(num_examples):

            curr_dataset = extract_images_from_dataset(ref_dataset, curr_images)
            curr_ontology = FewShotOntology(curr_dataset)

            # viz_dataset(f"greedies/{j}/train.png",curr_dataset)

            model = make_model(curr_ontology)

            # find worst-performing image of a few

            worst_score = -math.inf if metric.direction() == MetricDirection.LOWER_IS_BETTER else math.inf
            worst_img = None

            running_score = 0
            running_count = 0

            remaining_imgs = [img for img in positive_examples if img not in curr_images]

            pred_dets = {}

            for img in remaining_imgs[:max_test_imgs]:
                dets = model.predict(ref_dataset.images[img])
                pred_dets[img] = dets
                score = metric.evaluate_detections(ref_dataset.annotations[img], dets)

                running_score += score
                running_count += 1

                if (metric.direction() == MetricDirection.LOWER_IS_BETTER and score > worst_score) or (metric.direction() == MetricDirection.HIGHER_IS_BETTER and score < worst_score):
                    worst_score = score
                    worst_img = img
            
            if worst_img is None:
                break

            avg_score = running_score / running_count

            pred_dataset = DetectionDataset(
                classes=ref_dataset.classes,
                images={img_name: ref_dataset.images[img_name] for img_name in pred_dets},
                annotations=pred_dets
            )

            # viz_dataset(f"greedies/{j}/infer.png",pred_dataset)
            print(f"Score: {avg_score}")

            with open(f"greedies/{j}/metadata.json","w") as f:
                json.dump({
                    "score": avg_score,
                    "metric": metric.name(),
                    "images": curr_images,
                },f)

            curr_images.append(worst_img)
        
        ontologies_scores.append((
            FewShotOntology(extract_images_from_dataset(ref_dataset, curr_images)),
            avg_score
        ))
    
    best_ontology, best_score = max(ontologies_scores, key=lambda x: x[1])
    return best_ontology
