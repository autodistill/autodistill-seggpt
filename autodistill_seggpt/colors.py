from math import floor
from random import random

import numpy as np

palette_registry = {}


# use white
def next_white():
    return np.asarray([255, 255, 255])


white_palette = next_white()[None, ...]

palette_registry["white"] = (white_palette, next_white, "semantic")

# use r/g/b
rgb_palette = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]])


def next_rgb():
    global curr_idx
    ret = rgb_palette[curr_idx]
    curr_idx = (curr_idx + 1) % len(rgb_palette)
    return np.asarray(list(ret))

palette_registry["rgb"] = (rgb_palette, next_rgb, "instance")

class_colors = []
class_palette = np.ndarray((0,3),dtype=np.uint8)

def next_semantic_color(class_id:int):
    if class_id >= len(class_colors):
        # fill in the palette
        for i in range(len(class_colors),class_id+1):
            class_colors.append(np.array([floor(random()*255),floor(random()*255),floor(random()*255)]))
        # construct new class palette
        global class_palette
        class_palette = np.stack(class_colors,axis=0)
    return class_colors[class_id]

palette_registry["semantic"] = (class_palette, next_semantic_color, "semantic")

# choose your preset
preset = "semantic"

palette, next_color, seg_type = palette_registry[preset]