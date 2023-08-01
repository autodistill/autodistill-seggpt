from supervision import DetectionDataset
from autodistill.detection import DetectionBaseModel,DetectionOntology
import numpy as np
from tqdm import tqdm

# Run a BaseModel on every image in a dataset and return a new dataset with the predictions.
# Note: in an ideal world, this would eventually become a DetectionBaseModel method.
# But notice that it requires model.predict() to take in an np.ndarray--but the existing DetectionBaseModel.predict() takes in a filename.
def label_dataset(dataset:DetectionDataset,model:DetectionBaseModel)->DetectionDataset:
    if len(dataset.images)==0:
        # copy dataset
        return DetectionDataset(
            classes=[*dataset.classes],
            images={},
            annotations={},
        )

    # check if any images have masks
    dataset_has_masks = any(dataset.annotations[image_name].mask is not None for image_name in dataset.images)

    # get ontology of model--this determines pred_dataset.classes
    pred_classes = model.ontology.classes()

    # now label all images in dataset
    pred_annotations = {}

    for img_name,img in tqdm(dataset.images.items()):
        detections = model.predict(img)

        if dataset_has_masks and detections.mask is None:
            detections.mask = np.zeros((len(detections),img.shape[0],img.shape[1]),dtype=np.uint8)
        pred_annotations[img_name] = detections
    
    pred_dataset = DetectionDataset(
        classes=pred_classes,
        images=dataset.images,
        annotations=pred_annotations,
    )

    return pred_dataset

from random import sample

# Pick a random k example images for each target class
def shrink_dataset_to_size(dataset:DetectionDataset,max_imgs:int=15)->DetectionDataset:
    imgs = list(dataset.images.keys())

    if len(imgs) <= max_imgs: return dataset
    
    imgs = sample(imgs,max_imgs)

    new_images = {img_name:dataset.images[img_name] for img_name in imgs}
    new_annotations = {img_name:dataset.annotations[img_name] for img_name in imgs}

    return DetectionDataset(
        classes=dataset.classes,
        images=new_images,
        annotations=new_annotations
    )

import supervision as sv
import numpy as np

from .few_shot_ontology import FewShotOntology

# This acts like an oracle for a given FewShotOntology.
def extract_classes_from_dataset(old_dataset:DetectionDataset,ontology:FewShotOntology)->DetectionDataset:

    classes = ontology.classes()

    new_annotations = {}
    for img_name,detections in old_dataset.annotations.items():
        new_detectionss = []
        for new_class_id,cls in enumerate(classes):
            prompt = ontology.classToPrompt(cls)
            prompt_name,_ = prompt
            class_id = int(prompt_name.split("-")[0])

            new_detections = detections[detections.class_id==class_id]
            new_detections.class_id = np.ones_like(new_detections.class_id)*new_class_id

            new_detectionss.append(new_detections)
        new_annotations[img_name] = sv.Detections.merge(new_detectionss)
    
    return sv.DetectionDataset(
        classes=classes,
        images=old_dataset.images,
        annotations=new_annotations
    )
            
        