from autodistill_seggpt import SegGPT,find_best_examples,FewShotOntology
from autodistill.detection import CaptionOntology
import supervision as sv


#
# download dataset with a few labelled images (5-10 is recommended, but you can go as high as you like)
#
from roboflow import login,Roboflow

login()
rf = Roboflow()

project = rf.workspace("roboflow-4rfmv").project("climbing-y56wy")
dataset = project.version(5).download("coco-segmentation")

climbing_dataset = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
    force_masks=True
)

#
# Create ontology
#
climbing_ontology = CaptionOntology({
    "2-floor":"floor",
    "1-climbing-holds":"hold",
    "3-person":"climber",
})

# "Train" your ontology to use the best possible visual prompts
best_examples = find_best_examples(
    climbing_dataset,
    SegGPT
)

few_shot_ontology = FewShotOntology.from_examples(
    ref_dataset=climbing_dataset,
    ontology=climbing_ontology,
    examples=best_examples
)

base_model = SegGPT(
    ontology=few_shot_ontology,
    refine_detections=True
)

base_model.label("./unlabelled-climbing-photos", extension=".jpg")