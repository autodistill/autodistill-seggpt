# convert supervision Dataset into a torch Dataset

# split by class--for every class in every image, target output is the combined bitmask.

from torch.utils.data import Dataset
from supervision import DetectionDataset, Detections

from typing import Union, List, Iterable

import numpy as np

from dataclasses import dataclass
@dataclass
class Prompt:
    img: np.ndarray # rgb image
    mask: np.ndarray # rgb mask



class SegGPTDataset(Dataset):
    def __init__(self,sv_dataset:DetectionDataset):
        self.sv_dataset = sv_dataset
        promptss = [list(self.detections_to_prompts(sv_dataset.images[img_name],detections)) for img_name,detections in sv_dataset.annotations.items()]
        # flatten self.promptss
        self.prompts = [prompt for prompts in promptss for prompt in prompts]
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self,idx):
        
    def detections_to_prompts(self, img: np.ndarray, detections: Detections) -> Union[List[Prompt], Iterable[Prompt]]:
        raise NotImplementedError
    