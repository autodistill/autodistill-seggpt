# Given a dataset of a few images, find the best training examples from it.
# An ontology has k example images for each class.
# Try a bunch of random k-example-image-sets for each class, then pick the one which generalizes best to the rest of the images.
# Then combine those best-performing k-example-image-sets into a single ontology.
import math
from random import sample
from typing import List, Type, Union, Callable, Tuple

import numpy as np
import supervision as sv
from autodistill.detection import DetectionBaseModel
from supervision.dataset.core import DetectionDataset
from tqdm import tqdm

from .dataset_utils import (
    extract_classes_from_dataset,
    label_dataset,
    shrink_dataset_to_size,
)
from .few_shot_ontology import FewShotOntology
from .metrics import Metric, metrics_registry

# TODO: make multiple eval metrics. A metric could fit the interface get_score(gt_dataset,pred_dataset)->float,str.
# The float is the score, the str is a human-readable description of the score (plus some extra metadata like mAP-large, etc.)
# Metrics could include: mask AP, mask IoU, box AP, etc.
def use_all_examples(ref_dataset: DetectionDataset):
    # create few-shot ontologies for each class.
    cls_names = [f"{i}-{cls_name}" for i, cls_name in enumerate(ref_dataset.classes)]

    examples = {}
    for cls_name in cls_names:
        examples[cls_name] = []

    for img_name, detections in ref_dataset.annotations.items():
        # get unique classes in this image
        classes = np.unique(detections.class_id)

        for i in classes.tolist():
            cls_name = cls_names[i]
            examples[cls_name].append(img_name)
    return examples


def default_model(ontology: FewShotOntology) -> DetectionBaseModel:
    from .seggpt import SegGPT

    return SegGPT(
        ontology=ontology,
        refine_detections=False, # disable this since we don't use SAM feature map caching (or MobileSAM) yet. So SAM slows us down big-time.
    )


def find_best_examples(
    ref_dataset: DetectionDataset,
    model_class: Callable[[FewShotOntology], DetectionBaseModel],
    num_examples: int = 4,
    num_trials: int = 1,
    max_test_imgs: int = 10,
    which_metric: Union[str, Metric] = "iou",
):
    print("num_examples",num_examples)
    best_examples = {}

    if type(which_metric) == str:
        which_metric = metrics_registry[which_metric]
    metric, metric_name, metric_direction = which_metric

    for i, cls in enumerate(ref_dataset.classes):

        cls_deduped = f"{i}-{cls}"

        # get best example set for this class
        examples_scores = []

        gt_dataset = extract_classes_from_dataset(ref_dataset, [i])

        positive_examples = [
            img_name
            for img_name, detections in gt_dataset.annotations.items()
            if len(detections) > 0
        ]

        # TODO use CLIP to find a small AND diverse valid set.
        gt_dataset = shrink_dataset_to_size(gt_dataset, max_test_imgs)

        if len(positive_examples) == 0:
            best_examples[cls_deduped] = []
            continue
        
        curr_num_examples = min(num_examples, len(positive_examples))

        num_perms = perm(len(positive_examples), curr_num_examples)

        num_iters = min(num_perms, num_trials)
        combo_hashes = range(num_perms)
        sub_combo_hashes = sample(combo_hashes, num_iters)

        print(f"Finding best examples for class {cls_deduped}.")

        combo_pbar = tqdm(sub_combo_hashes)
        max_score = -math.inf

        for combo_hash in combo_pbar:
            image_choices = combo_hash_to_choices(
                combo_hash, positive_examples, curr_num_examples
            )
            onto_tuples = [((cls_deduped, image_choices), cls)]

            ontology = FewShotOntology(ref_dataset, onto_tuples)
            model = make_model(ontology)  # model must take only an Ontology as a parameter
            pred_dataset = label_dataset(gt_dataset, model)
            score = metric(gt_dataset, pred_dataset).tolist()

            examples_scores.append((image_choices, score))
            max_or_min = max if metric_direction == 1 else min
            max_score = max_or_min(max_score, score)
            combo_pbar.set_description(f"Best/Last {metric_name}: {round(max_score,2)}/{round(score,2)}")

        my_best_examples, best_score = max(examples_scores, key=lambda x: x[1])
        best_examples[cls_deduped] = my_best_examples
    return best_examples


def combo_hash_to_choices(
    fact: int, candidates: List[any], num_choices: int
) -> List[any]:
    assert len(candidates) >= num_choices, "Not enough candidates to choose from."

    chosen = []
    curr_fact = fact
    remaining_candidates = [*candidates]
    for i in range(num_choices, 0, -1):
        which_remaining_candidate = curr_fact % i
        chosen.append(remaining_candidates.pop(which_remaining_candidate))
        curr_fact = curr_fact // i
    return chosen


import math

def perm(n, k):
    return int(math.factorial(n) / math.factorial(n - k))

def choose(n, k):
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))

Imgset = List[str]
imgset_entries: List[Tuple[Imgset,DetectionDataset]] = []

def find_best_ensembles(
    ref_dataset: DetectionDataset,
    valid_dataset: DetectionDataset,
    make_model: Callable[[FewShotOntology], DetectionBaseModel]=default_model,
    num_examples: int = 5,
    trials_per_size: int = 5,
    max_test_imgs: int = 10,
    which_metric: Union[str, Metric] = "iou",
):
    best_ensembles = {}

    if type(which_metric) == str:
        which_metric = metrics_registry[which_metric]
    metric, metric_name, metric_direction = which_metric

    for i,cls in enumerate(ref_dataset.classes):
        print(f"Finding best ensemble for class {i}-{cls}.")

        gt_dataset = extract_classes_from_dataset(ref_dataset, [i])

        positive_examples = [
            img_name
            for img_name, detections in gt_dataset.annotations.items()
            if len(detections) > 0
        ]

        if len(positive_examples) == 0:
            best_ensembles[i] = []
            continue

        # TODO use CLIP to find a small AND diverse valid set.
        small_valid_dataset = extract_classes_from_dataset(valid_dataset, [i])
        small_valid_dataset = shrink_dataset_to_size(small_valid_dataset, max_test_imgs)
        
        # an Ensemble is a set of N imgsets of M imgs.

        # Each M examples are used together to generate a predicted mask.
        # The N predicted masks are then merged together to form the final prediction.

        # We will make atomic Ensembles of size 1, then try merging their predictions together.
        # Note: this is a greedy algorithm, not an optimal one. 

        global imgset_entries
        # imgset_entries = []

        def infer_imgset(imgset:List[str]):
            # look up imgset in imgset_entries
            # if it's not there, create it and add it to imgset_entries
            # then return the dataset
            for entry in imgset_entries:
                if entry[0] == imgset:
                    return entry[1]

            onto_tuples = [((f"{i}-{cls}",imgset), cls)]
            ontology = FewShotOntology(ref_dataset, onto_tuples)
            model = make_model(ontology)
            pred_dataset = label_dataset(small_valid_dataset, model)

            imgset_entries.append((imgset,pred_dataset))
            return pred_dataset

        entries:List[Tuple[List[Imgset],float]] = []

        best_score = -math.inf if metric_direction == 1 else math.inf
        def ensemble_to_entry(ensemble:List[Imgset]):
            for entry in entries:
                if entry[0] == ensemble:
                    return entry

            # assume imgset_entries has already been set with each imgset
            datasets = [infer_imgset(imgset) for imgset in ensemble]

            # then merge detections + nms them
            new_annotations = {}
            for img_name in small_valid_dataset.annotations:
                all_detections = []
                for dataset in datasets:
                    all_detections.append(dataset.annotations[img_name])
                merged = sv.Detections.merge(all_detections).with_nms()
                new_annotations[img_name] = merged
            
            new_dataset = DetectionDataset(
                classes=small_valid_dataset.classes,
                images=small_valid_dataset.images,
                annotations=new_annotations
            )
            score = metric(small_valid_dataset, new_dataset).tolist()

            nonlocal best_score
            if (score > best_score and metric_direction == 1) or (score < best_score and metric_direction == -1):
                best_score = score

                assert pbar is not None, "pbar is None"
                pbar.set_description(f"Best {metric_name}: {best_score:.2f}")

            entry = ensemble, score
            entries.append(entry)

            ret = ensemble, new_dataset,score
            return ret
        
        def sort_entries():
            entries.sort(key=lambda x: x[1],reverse=metric_direction==1)
        
        max_singletons = 7
        max_singletons = min(max_singletons,len(positive_examples))
        print("making singletons.")
        pbar = tqdm(sample(positive_examples,max_singletons))

        # make all singletons
        for img_name in pbar:
            ensemble = [[img_name]]
            ensemble_to_entry(ensemble)
        sort_entries()
        
        # create imgsets with 2 examples, using the top-k singletons

        print("making 2-image pairs.")

        K = 3
        Q = 3
        pbar = tqdm(entries[:K])

        for j,(ensemble,score) in enumerate(pbar):
            q = min(Q,len(entries)-j-1)

            good_img_name = ensemble[0][0]
            to_pair_with = list(ref_dataset.annotations.keys())[j+1:]
            to_pair_with = sample(to_pair_with,q)

            for img_name in to_pair_with:
                new_ensemble = [[good_img_name,img_name]]
                ensemble_to_entry(new_ensemble)
        
        # now try merging these ensembles together!
        # we are no longer calling infer_imgset--we are only using the imgset_entries we already have


        K = 10
        Q = 10
        iters = 10

        print("merging ensembles.")
        pbar = tqdm(range(iters))

        for i in pbar:
            sort_entries()

            # take the top K entries
            # try merging them with other entries (Q per entry)

            for i,(ensemble,score) in enumerate(entries[:K]):
                q = min(Q,len(entries)-1-i)

                to_pair_with = sample(entries[i+1:],q)

                for ensemble_2,*_ in to_pair_with:
                    new_ensemble = ensemble + ensemble_2 # eliminate duplicate imgsets
                    ensemble_to_entry(new_ensemble)
        
        sort_entries()

        best_entry = entries[0]

        best_ensembles[cls] = best_entry[0]
    return best_ensembles