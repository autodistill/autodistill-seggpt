from __future__ import annotations

from supervision.dataset.core import DetectionDataset
from supervision import Detections

from autodistill.core import Ontology
from autodistill.detection import CaptionOntology,DetectionOntology,DetectionBaseModel

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np



@dataclass
class FewShotOntology(DetectionOntology):
    def __init__(self,
                 ref_dataset:DetectionDataset,
                 # each tuple in the list has form:
                 # ( (training_class_name, [reference_image_ids]), output_class_name )]))
                 # i.e. ( ("1-climbing-holds",["demo-holds-1.jpg","demo-holds-2.jpg"]), "climbing-hold" )
                 ontology: List[Tuple[
                        Tuple[str,List[str]],
                        str
                     ]]
        ):
        self.ref_dataset = ref_dataset
        self.ontology = ontology
        rich_ontology = self.enrich_ontology(ontology)
        self.rich_ontology = rich_ontology
    
    
    def prompts(self)->List[Tuple[str,List[str]]]:
        return [key for key,val in self.ontology]
    def classes(self)->List[str]:
        return [val for key,val in self.ontology]
    def promptToClass(self,prompt:str)->str:
        for key,val in self.ontology:
            if key == prompt:
                return val
        raise Exception("No class found for prompt.")
    def classToPrompt(self,cls:str)->str:
        for key,val in self.ontology:
            if val == cls:
                return key
        raise Exception("No prompt found for class.")

    # my custom, non-DetectionOntology methods
    def rich_prompts(self)->List[List[Tuple[np.ndarray,Detections]]]:
        return [key for key,val in self.rich_ontology]
    def rich_prompt_to_class(self,rich_prompt:List[Tuple[np.ndarray,Detections]])->str:
        for key,val in self.rich_ontology:
            if key == rich_prompt:
                return val
        raise Exception("No class found for prompt.")

    # using lists-of-pairs instead of dicts:
    def enrich_ontology(self, ontology: List[Tuple[
                        Tuple[str,List[str]],
                        str
                        ]]
        )->List[Tuple[List[Tuple[np.ndarray,Detections]],str]]:

        rich_ontology = []

        for basic_key,val in ontology:
            cls_name, ref_img_names = basic_key

            cls_names = [f"{i}-{cls_name}" for i,cls_name in enumerate(self.ref_dataset.classes)]
            cls_id = cls_names.index(cls_name)

            new_key = []
            for ref_img_name in ref_img_names:
                detections = self.ref_dataset.annotations[ref_img_name]
                detections = detections[detections.class_id==cls_id]
                image = self.ref_dataset.images[ref_img_name]
                new_key.append((image,detections))
            rich_ontology.append((new_key,val))
        return rich_ontology


    @staticmethod
    def from_examples(
            ref_dataset:DetectionDataset,
            ontology:CaptionOntology,
            examples: Dict[str,List[str]]
    ) -> FewShotOntology:
        onto_tuples = []
        for prompt,examples in examples.items():
            if prompt not in ontology.prompts():
                continue
            cls = ontology.promptToClass(prompt)
            onto_tuples.append(
                ((prompt,examples),cls)
            )
        return FewShotOntology(ref_dataset,onto_tuples)
