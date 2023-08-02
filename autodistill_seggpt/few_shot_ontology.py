from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from autodistill.core import Ontology
from autodistill.detection import CaptionOntology, DetectionBaseModel, DetectionOntology
from supervision import Detections
from supervision.dataset.core import DetectionDataset

# TODO turn FewShotOntology into a thin wrapper around DetectionDataset
# move the searching/etc. logic into find_best_examples.
# Maybe also try to make nice/quick serialization for shareability.

def default_model(ontology: FewShotOntology) -> DetectionBaseModel:
    from .seggpt import SegGPT

    return SegGPT(
        ontology=ontology,
        refine_detections=False, # disable this since we don't use SAM feature map caching (or MobileSAM) yet. So SAM slows us down big-time.
    )

@dataclass
class FewShotOntology(DetectionOntology):
    def __init__(
        self,
        ref_dataset: DetectionDataset,
        # each tuple in the list has form:
        # ( (training_class_name, [reference_image_ids]), output_class_name )]))
        # i.e. ( ("1-climbing-holds",["demo-holds-1.jpg","demo-holds-2.jpg"]), "climbing-hold" )
        ontology: List[Tuple[Tuple[str, List[str]], str]] = None,
    ):
        self.ref_dataset = ref_dataset

        if ontology is None:
            # make sensible defaults.
            # basically: turn the class list ["climbing-hold","climbing-hold","floor","climber"]
            # into the ontology: { "1-climbing-hold": "climbing-hold", "2-floor": "floor", "3-climber": "climber"}

            # This handles a common case where class idx 0 is empty and shares a name with other class idxes.

            class_name_to_id = {}

            for detections in ref_dataset.annotations.values():
                # get unique classes in this image
                classes = np.unique(detections.class_id)
                for i in classes.tolist():
                    class_name = ref_dataset.classes[i]
                    class_name_to_id[class_name] = i

            ontology = CaptionOntology(
                {
                    f"{i}-{cls_name}": cls_name
                    for cls_name, i in class_name_to_id.items()
                }
            )

            from .find_best_examples import find_best_examples

            best_examples = find_best_examples(ref_dataset, default_model)
            print("best_examples", best_examples)
            ontology = FewShotOntology.examples_to_tuples(ontology, best_examples)

        self.ontology = ontology
        rich_ontology = self.enrich_ontology(ontology)
        self.rich_ontology = rich_ontology

    # DetectionOntology methods
    def prompts(self) -> List[Tuple[str, List[str]]]:
        return [key for key, val in self.ontology]

    def classes(self) -> List[str]:
        return [val for key, val in self.ontology]

    def promptToClass(self, prompt: str) -> str:
        for key, val in self.ontology:
            if key == prompt:
                return val
        raise Exception("No class found for prompt.")

    def classToPrompt(self, cls: str) -> str:
        for key, val in self.ontology:
            if val == cls:
                return key
        raise Exception("No prompt found for class.")

    # my custom, non-DetectionOntology methods
    def rich_prompts(self) -> List[List[Tuple[np.ndarray, Detections]]]:
        return [key for key, val in self.rich_ontology]

    def rich_prompt_to_class(
        self, rich_prompt: List[Tuple[np.ndarray, Detections]]
    ) -> str:
        for key, val in self.rich_ontology:
            if key == rich_prompt:
                return val
        raise Exception("No class found for prompt.")

    # Turn filenames into images and detections
    def enrich_ontology(
        self, ontology: List[Tuple[Tuple[str, List[str]], str]]
    ) -> List[Tuple[List[Tuple[np.ndarray, Detections]], str]]:
        rich_ontology = []

        for basic_key, val in ontology:
            cls_name, ref_img_names = basic_key

            cls_names = [
                f"{i}-{cls_name}" for i, cls_name in enumerate(self.ref_dataset.classes)
            ]
            cls_id = cls_names.index(cls_name)

            new_key = []
            for ref_img_name in ref_img_names:
                detections = self.ref_dataset.annotations[ref_img_name]
                detections = detections[detections.class_id == cls_id]
                image = self.ref_dataset.images[ref_img_name]
                new_key.append((image, detections))
            rich_ontology.append((new_key, val))
        return rich_ontology

    @staticmethod
    def examples_to_tuples(
        ontology: CaptionOntology,
        examples: Dict[str, List[str]],
    ) -> List[Tuple[Tuple[str, List[str]], str]]:
        onto_tuples = []
        for prompt, examples in examples.items():
            if prompt not in ontology.prompts():
                continue
            cls = ontology.promptToClass(prompt)
            onto_tuples.append(((prompt, examples), cls))
        return onto_tuples

    @staticmethod
    def from_examples(
        ref_dataset: DetectionDataset,
        ontology: CaptionOntology,
        examples: Dict[str, List[str]],
    ) -> FewShotOntology:
        onto_tuples = FewShotOntology.examples_to_tuples(ontology, examples)
        return FewShotOntology(ref_dataset, onto_tuples)
