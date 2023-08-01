import cv2
import numpy as np
import supervision as sv
from segment_anything import SamPredictor, sam_model_registry

sam_res = (256, 256)

eps = 1e-6

from .metrics import get_combined_mask


def refine_detections(
    img: np.ndarray,
    detections: sv.Detections,
    predictor: SamPredictor,
    use_masks=True,
):
    # TODO use each detection mask (or bbox) as a prompt for SAM

    predictor.set_image(img)

    new_detections = []

    for detection in detections:
        det_box, det_mask, det_conf, det_cls, _ = detection
        og_h, og_w = det_mask.shape

        resized_mask = cv2.resize(det_mask, sam_res)

        mask_input = resized_mask[None, ...] if use_masks else None
        masks_np, iou_predictions, low_res_masks = predictor.predict(
            mask_input=mask_input, box=det_box[None, :], multimask_output=True
        )

        n, h, w = masks_np.shape

        # old strategy: single-mask output.
        # assert n == 1, f"n: {n}"
        # new strategy: pick mask with highest IoU with det_mask
        intersections = np.logical_and(masks_np, det_mask[None, ...]).sum(axis=(1, 2))
        unions = np.logical_or(masks_np, det_mask[None, ...]).sum(axis=(1, 2))
        ious = intersections / (unions + eps)

        best_idx = np.argmax(ious)

        (m,) = iou_predictions.shape
        assert m == n, f"m: {m}, n: {n}"

        mask = cv2.resize(masks_np[best_idx].astype(np.uint8), (og_h, og_w))
        mask = mask[None, ...]
        assert mask.shape == (1, og_h, og_w), f"mask.shape: {mask.shape}"

        class_id = np.array([det_cls])

        confidence = np.array([iou_predictions[best_idx]])

        new_det = sv.Detections(
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            xyxy=sv.detection.utils.mask_to_xyxy(mask),
        )
        new_detections.append(new_det)

    detections = sv.Detections.merge(new_detections)
    # set mask to empty if None
    if detections.mask is None:
        assert len(detections) == 0, f"len(detections): {len(detections)}"
        # get img shape
        h, w = img.shape[:2]
        detections.mask = np.zeros_like((0, h, w))
    return detections

def _load_sam(sam_type: str = "vit_h") -> SamPredictor:
    raise NotImplementedError("This doesn't handle any downloading! Use rf_segment_anything load_SAM instead.")
    if sam_type == "vit_h":
        sam_type, sam_ckpt = "vit_h", "weights/sam_vit_h_4b8939.pth"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif sam_type == "vit_t":
        sam_type, sam_ckpt = "vit_t", "weights/mobile_sam.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()

    sam.sam_type = sam_type

    predictor = SamPredictor(sam)

    return predictor
