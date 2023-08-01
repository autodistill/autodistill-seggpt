import numpy as np

from .colors import palette

eps = 1e-10


def quantize(img):
    global palette

    # quantize image to palette
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    cosine = np.sum(img[..., None, :] * palette[None, None, ...], axis=-1)
    cosine = cosine / (
        np.linalg.norm(img, axis=-1, keepdims=True)
        * np.linalg.norm(palette, axis=-1)[None, None, :]
        + eps
    )

    idx = np.argmax(cosine, axis=-1)

    # set black pixels as any pixel that is not close to any color in the palette
    black_dist = np.linalg.norm(img, axis=-1)
    idx[black_dist < 40] = len(palette)

    new_palette = np.concatenate([palette, [[0, 0, 0]]], axis=0)

    img = new_palette[idx]
    return img.astype("uint8")


from scipy.ndimage.measurements import label


def quantized_to_bitmasks(img, palette):
    # get "components" of each color
    filtered_components = []
    for color_idx in range(palette.shape[0]):
        color = palette[color_idx]
        matching_pixels = np.all(img == color, axis=-1)
        # now make components
        componentized, num_components = label(matching_pixels)

        for component_idx in range(1, num_components + 1):
            component = componentized == component_idx
            filtered_components.append(component)

    filtered_components = [
        (component * 255).astype("uint8") for component in filtered_components
    ]
    return filtered_components


import supervision as sv


def bitmasks_to_detections(bitmasks, catId):
    class_id = np.asarray([catId]).astype("int64")
    detections = [
        sv.Detections(
            xyxy=sv.detection.utils.mask_to_xyxy(bitmask[None, ...]),
            mask=bitmask[None, ...] > 0,
            confidence=np.ndarray([1]),
            class_id=class_id,
        )
        for bitmask in bitmasks
    ]
    return sv.Detections.merge(detections)
