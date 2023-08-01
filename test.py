from autodistill_seggpt import SegGPT,FewShotOntology


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

climbing_ontology = FewShotOntology(climbing_dataset)

base_model = SegGPT(
    ontology=climbing_ontology,
    refine_detections=True
)

base_model.label("./unlabelled-climbing-photos", extension=".jpg")