from math import floor
from random import random

import numpy as np

palette_registry = {}

class Coloring:
    @classmethod
    def type():
        raise NotImplementedError
    @classmethod
    def next_color(cls,class_id:int):
        raise NotImplementedError
    @classmethod
    def palette(cls):
        raise NotImplementedError

# color based on class ID
class SemanticColoring(Coloring):
    @classmethod
    def type(cls):
        return "semantic"
    @classmethod
    def next_color(cls,class_id:int):
        raise NotImplementedError
    @classmethod
    def palette(cls):
        raise NotImplementedError

class InstanceColoring(Coloring):
    @classmethod
    def type(cls):
        return "instance"
    @classmethod
    def next_color(cls,class_id:int):
        return cls._next_color(cls)
    @classmethod
    def _next_color(cls):
        raise NotImplementedError
    @classmethod
    def palette(cls):
        raise NotImplementedError

class PaletteSemanticColoring(SemanticColoring):
    class_palette = None
    @classmethod
    def next_color(cls,class_id:int):
        return cls.class_palette[class_id % len(cls.class_palette)]
    @classmethod
    def palette(cls):
        return cls.class_palette

class PaletteInstanceColoring(InstanceColoring):
    instance_palette = None
    curr_idx = 0
    @classmethod
    def _next_color(cls):
        ret = cls.instance_palette[cls.curr_idx % len(cls.instance_palette)]
        cls.curr_idx = (cls.curr_idx + 1) % len(cls.instance_palette)
        return np.asarray(list(ret))
    @classmethod
    def palette(cls):
        return cls.instance_palette

class White(PaletteSemanticColoring):
    class_palette = np.asarray([[255, 255, 255]])

palette_registry["white"] = White

class RGB(PaletteInstanceColoring):
    # use r/g/b
    instance_palette = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

palette_registry["rgb"] = RGB

class RGBSemantic(PaletteSemanticColoring):
    # use r/g/b
    class_palette = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

palette_registry["rgb_semantic"] = RGBSemantic

class RandomColors(SemanticColoring):
    class_colors = []
    class_palette = np.ndarray((0,3),dtype=np.uint8)
    @classmethod
    def next_color(cls,class_id:int):
        if class_id >= len(cls.class_colors):
            # fill in the palette
            for i in range(len(cls.class_colors),class_id+1):
                cls.class_colors.append(np.array([floor(random()*255),floor(random()*255),floor(random()*255)]))
            # construct new class palette
            cls.class_palette = np.stack(cls.class_colors,axis=0)
        return cls.class_colors[class_id]
    @classmethod
    def palette(cls):
        return cls.class_palette

palette_registry["random"] = RandomColors

# choose your preset
preset = "rgb_semantic"

color = palette_registry[preset]