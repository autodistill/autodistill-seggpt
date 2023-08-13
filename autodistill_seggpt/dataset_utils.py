import numpy as np
from autodistill.detection import DetectionBaseModel, DetectionOntology
from supervision import DetectionDataset
from tqdm import tqdm


# Note: in an ideal world, this would eventually become a DetectionBaseModel method.
# But notice that it requires model.predict() to take in an np.ndarray--but the existing DetectionBaseModel.predict() takes in a filename.
def label_dataset(
    dataset: DetectionDataset, model: DetectionBaseModel, use_tqdm: bool = False
) -> DetectionDataset:
    """
    Run a BaseModel on every image in a dataset and return a new dataset with the predictions.

    Keyword arguments:
    dataset -- the dataset to label
    model -- the base model which will label the dataset. Note: must support predict(img:np.ndarray).
    use_tqdm -- whether to show a progress bar
    """
    if len(dataset.images) == 0:
        # copy dataset
        return DetectionDataset(
            classes=[*dataset.classes],
            images={},
            annotations={},
        )

    # check if any images have masks
    dataset_has_masks = any(
        dataset.annotations[image_name].mask is not None
        for image_name in dataset.images
    )

    # get ontology of model--this determines pred_dataset.classes
    pred_classes = model.ontology.classes()

    # now label all images in dataset
    pred_annotations = {}

    itr = tqdm(dataset.images.items()) if use_tqdm else dataset.images.items()

    for img_name, img in itr:
        detections = model.predict(img)

        if dataset_has_masks and detections.mask is None:
            detections.mask = np.zeros(
                (len(detections), img.shape[0], img.shape[1]), dtype=np.uint8
            )
        pred_annotations[img_name] = detections

    pred_dataset = DetectionDataset(
        classes=pred_classes,
        images=dataset.images,
        annotations=pred_annotations,
    )

    return pred_dataset


from random import sample


def shrink_dataset_to_size(
    dataset: DetectionDataset, max_imgs: int = 15
) -> DetectionDataset:
    """
    Pick a random subset of the dataset.

    Keyword arguments:
    dataset -- the dataset to shrink
    max_imgs -- the maximum number of images to keep in the dataset
    """
    imgs = list(dataset.images.keys())

    if len(imgs) <= max_imgs:
        # copy dataset
        return DetectionDataset(
            classes=dataset.classes,
            images={*dataset.images},
            annotations={*dataset.annotations},
        )

    imgs = sample(imgs, max_imgs)

    new_images = {img_name: dataset.images[img_name] for img_name in imgs}
    new_annotations = {img_name: dataset.annotations[img_name] for img_name in imgs}

    return DetectionDataset(
        classes=dataset.classes, images=new_images, annotations=new_annotations
    )


from typing import List

import numpy as np
import supervision as sv

from .few_shot_ontology import OldFewShotOntology


# This acts like an oracle for a given FewShotOntology.
def extract_classes_from_dataset(
    old_dataset: DetectionDataset, class_ids: List[int]
) -> DetectionDataset:
    """
    Extract a subset of classes from a dataset.
    This re-maps the class_ids to be contiguous.

    Keyword arguments:
    old_dataset -- the dataset to extract from
    class_ids -- the class_ids (as integers) to extract
    """

    new_annotations = {}
    for img_name, detections in old_dataset.annotations.items():
        new_detectionss = []
        for new_class_id, class_id in enumerate(class_ids):
            new_detections = detections[detections.class_id == class_id]
            new_detections.class_id = (
                np.ones_like(new_detections.class_id) * new_class_id
            )

            new_detectionss.append(new_detections)
        new_annotations[img_name] = sv.Detections.merge(new_detectionss)

    classes = [old_dataset.classes[class_id] for class_id in class_ids]
    return sv.DetectionDataset(
        classes=classes, images=old_dataset.images, annotations=new_annotations
    )

def extract_images_from_dataset(dataset: DetectionDataset, images: List[str])->DetectionDataset:
    return DetectionDataset(
        classes=dataset.classes,
        images={img_name:dataset.images[img_name] for img_name in images},
        annotations={img_name:dataset.annotations[img_name] for img_name in images},
    )