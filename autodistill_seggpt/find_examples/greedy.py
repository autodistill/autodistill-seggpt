import math
from random import sample
from typing import List, Type, Union, Callable

import numpy as np
import supervision as sv
from autodistill.detection import DetectionBaseModel
from supervision.dataset.core import DetectionDataset
from tqdm import tqdm

from ..dataset_utils import (
    extract_classes_from_dataset,
    label_dataset,
    shrink_dataset_to_size,
)
from ..few_shot_ontology import FewShotOntologySimple
from ..metrics import Metric, MetricDirection

from random import choice
def grow_ontology(
    # general params
    ref_dataset: DetectionDataset,
    make_model: Callable[[FewShotOntologySimple], DetectionBaseModel],
    metric: Metric,

    # sample-specific params
    num_examples: int = 4,
    num_trials: int = 2,
    max_test_imgs: int = 3,
)->FewShotOntologySimple:

    if len(ref_dataset.images) == 0:
        return ref_dataset

    valid_dataset = shrink_dataset_to_size(ref_dataset, max_test_imgs)
    num_examples = min(num_examples, len(ref_dataset.images))

    ontologies_scores = []

    for _ in range(num_trials):
        curr_images = [choice(list(ref_dataset.images.keys()))]

        avg_score = 0

        for i in range(num_examples):

            curr_dataset = extract_classes_from_dataset(ref_dataset, curr_images)
            curr_ontology = FewShotOntologySimple(curr_dataset)

            model = make_model(curr_ontology)

            # find worst-performing image of a few

            worst_score = math.inf if metric.direction() == MetricDirection.LOWER_IS_BETTER else -math.inf
            worst_img = None

            running_score = 0
            running_count = 0

            remaining_imgs = [img for img in ref_dataset.images.keys() if img not in curr_images]

            for img in remaining_imgs[:max_test_imgs]:
                dets = model.predict(ref_dataset.images[img])
                score = metric.evaluate_detections(ref_dataset.annotations[img], dets)

                running_score += score
                running_count += 1

                if (metric.direction() == MetricDirection.LOWER_IS_BETTER and score < worst_score) or (metric.direction == MetricDirection.HIGHER_IS_BETTER and score > worst_score):
                    worst_score = score
                    worst_img = img
            
            if worst_img is None:
                break

            avg_score = running_score / running_count

            curr_images.append(worst_img)
        
        ontologies_scores.append((
            extract_classes_from_dataset(ref_dataset, curr_images),
            avg_score
        ))
    
    best_ontology, best_score = max(ontologies_scores, key=lambda x: x[1])
    return best_ontology
