from ..dataset_utils import extract_classes_from_dataset
from ..metrics import Metric
from ..few_shot_ontology import FewShotOntology,SeparatedFewShotOntology
from typing import Callable, Dict, List

from supervision import DetectionDataset
from autodistill import DetectionBaseModel

def separable(ontology_finding_fn):
    def wrapper(
            ref_dataset: DetectionDataset,
            make_model: Callable[[FewShotOntology], DetectionBaseModel],
            metric: Metric,
            *args, separate=False, **kwargs):
        if not separate:
            return ontology_finding_fn(
                ref_dataset, make_model, metric, *args, **kwargs
            )
        else:

            ref_datasets = {}

            # get many sub-datasets
            for i in range(len(ref_dataset.classes)):
                sub_dataset = extract_classes_from_dataset(ref_dataset, [i])

                sub_ontology = ontology_finding_fn(sub_dataset, make_model, metric, *args, **kwargs)

                ref_datasets[i] = sub_ontology.ref_dataset
            
            return SeparatedFewShotOntology(ref_datasets)

    return wrapper