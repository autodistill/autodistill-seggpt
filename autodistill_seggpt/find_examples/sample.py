
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
from ..metrics import Metric, metrics_registry, metric_on_detections

# TODO: make multiple eval metrics. A metric could fit the interface get_score(gt_dataset,pred_dataset)->float,str.
# The float is the score, the str is a human-readable description of the score (plus some extra metadata like mAP-large, etc.)
# Metrics could include: mask AP, mask IoU, box AP, etc.
from random import choice
def sample_ontology(
    # general params
    ref_dataset: DetectionDataset,
    make_model: Callable[[FewShotOntologySimple], DetectionBaseModel],
    metric: Metric,

    # sample-specific params
    num_examples: int = 4,
    num_trials: int = 1,
    max_test_imgs: int = 10,
)->FewShotOntologySimple:
    print("num_examples",num_examples)

    positive_examples = [
        img_name
        for img_name, detections in ref_dataset.annotations.items()
        if len(detections) > 0
    ]

    if len(positive_examples) == 0:
        return ref_dataset

    # TODO use CLIP to find a small AND diverse valid set.
    valid_dataset = shrink_dataset_to_size(ref_dataset, max_test_imgs)

    num_examples = min(num_examples, len(positive_examples))
    num_combos = perm(len(positive_examples), num_examples)

    num_iters = min(num_combos, num_trials)

    combo_hashes = range(num_combos)
    sub_combo_hashes = sample(combo_hashes, num_iters)

    combo_pbar = tqdm(sub_combo_hashes)
    max_score = -math.inf

    ontologies_scores = []

    for combo_hash in combo_pbar:
        image_choices = combo_hash_to_choices(
            combo_hash, positive_examples, num_examples
        )
        sub_dataset = DetectionDataset(
            classes=ref_dataset.classes,
            images={img_name: ref_dataset.images[img_name] for img_name in image_choices},
            annotations={img_name: ref_dataset.annotations[img_name] for img_name in image_choices},
        )
        ontology = FewShotOntologySimple(sub_dataset)
        model = make_model(ontology)  # model must take only an Ontology as a parameter
        pred_dataset = label_dataset(valid_dataset, model)
        score = metric(valid_dataset, pred_dataset).tolist()

        ontologies_scores.append((ontology, score))
        max_or_min = max if metric.direction == 1 else min
        max_score = max_or_min(max_score, score)
        combo_pbar.set_description(f"Best/latest {metric.name}: {round(max_score,2)}/{round(score,2)}")

    best_ontology, best_score = max(ontologies_scores, key=lambda x: x[1])

    return best_ontology

from combinadics import Combination

# use the Macaulay integer representation of combinations to choose a subset of images
# We use this since unique integers map to unique combinations - by taking a random sample of ints, we can get a non-repeating set of Ontologies.
def combo_hash_to_choices(
    hash: int, candidates: List[any], num_choices: int
) -> List[any]:
    assert len(candidates) >= num_choices, "Not enough candidates to choose from."

    combination = Combination(len(candidates), num_choices)
    idxes = combination.Element(hash).data

    chosen = []
    for idx in idxes:
        chosen.append(candidates[idx])
    return chosen

import math
def perm(n, k):
    return int(math.factorial(n) / math.factorial(n - k))

def choose(n, k):
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))