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

# choose your preset
preset = "white"

palette, next_color, seg_type = palette_registry[preset]
