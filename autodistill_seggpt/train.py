"""
Structure:
Dataset logic - converts an SV dataset into a torch dataset of images and RGB targets
Model logic - takes in a prompt img, prompt mask, and infer img (not preprocessed, just on cuda and still in BGR format) and outputs an RGB img
Loss fns - gives L1 loss between RGB target and pred.
Training loop - has optimizer, runs for certain # of iters, etc.
Conversion step - converts RGB targets into Detections
"""

# dataset

import torch
from torch.utils.data import Dataset

from supervision import DetectionDataset
from .colors import Coloring
from .seggpt import SegGPT


class SegGPTTorchDataset(Dataset):
    def __init__(self, sv_dataset: DetectionDataset,coloring: Coloring,device: str):
        self.sv_dataset = sv_dataset
        self.img_names = list(sv_dataset.images.keys())
        self.coloring = coloring
        self.device = device
    def __len__(self):
        return len(self.sv_dataset.images)
    def __getitem__(self,idx):
        # make GT mask
        img_name = self.img_names[idx]
        img = self.sv_dataset.images[img_name]

        detections = self.sv_dataset.annotations[img_name]
        img,gt_mask = SegGPT.prepare_ref_img(img,detections,self.coloring)

        # convert to torch
        img = torch.from_numpy(img).float().to(self.device)
        gt_mask = torch.from_numpy(gt_mask).float().to(self.device)

        return img,gt_mask

# model
from .seggpt import \
    ckpt_path, model, device, \
    res, hres, \
    imagenet_mean, imagenet_std

from seggpt_inference import prepare_model
from torch import nn
from torch.nn import functional as F

import numpy as np

preprocess = False

class SegGPTTorchModel(nn.Module):
    def __init__(self,coloring: Coloring):
        super().__init__()

        self.coloring = coloring

        seg_type = self.coloring.type()
        self.model = prepare_model(ckpt_path, model, seg_type).to(device)
        
        # reference img
        self.reference_img = nn.Parameter(torch.zeros((1,res,hres,3),dtype=torch.float32,device=device))
        self.reference_mask = nn.Parameter(torch.zeros((1,res,hres,3),dtype=torch.float32,device=device))

        # kaiming init
        nn.init.kaiming_normal_(self.reference_img)
        nn.init.kaiming_normal_(self.reference_mask)

        self.imagenet_mean = torch.tensor(imagenet_mean).to(device)
        self.imagenet_std = torch.tensor(imagenet_std).to(device)

    def preprocess(self,img):
        img = img / 255.0
        img = img - self.imagenet_mean
        img = img / self.imagenet_std
        return img
    
    def postprocess(self,img):
        img = img * self.imagenet_std
        img = img + self.imagenet_mean
        img = img * 255.0
        return img
    
    def run_one_image(self,img, tgt, model, device):
        x = img
        # make it a batch-like
        x = torch.einsum('nhwc->nchw', x)

        tgt = tgt
        # make it a batch-like
        tgt = torch.einsum('nhwc->nchw', tgt)

        bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
        bool_masked_pos[model.patch_embed.num_patches//2:] = 1
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
        valid = torch.ones_like(tgt)

        if self.model.seg_type == 'instance':
            seg_type = torch.ones([valid.shape[0], 1])
        else:
            seg_type = torch.zeros([valid.shape[0], 1])
        
        # this doesn't matter since we're only using one example image at a time - and one batch has only one example. :(
        feat_ensemble = 0 if len(x) > 1 else -1

        torch.save(x, 'x.pt')
        torch.save(tgt, 'tgt.pt')

        _, y, mask = model(x,tgt, bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y)

        output = y[0, y.shape[1]//2:, :, :]

        output = self.postprocess(output)
        torch.save(output, 'output.pt')

        return output

    def forward(self,img):
        # prepare ref img/mask from reference_img/reference_mask

        if preprocess:
            ref_imgs = self.preprocess(self.reference_img)
            ref_masks = self.preprocess(self.reference_mask)
        else:
            ref_imgs = self.reference_img
            ref_masks = self.reference_mask

        assert len(img.shape) == 3, f"img.shape: {img.shape}"

        # convert ref_imgs from (N,H,W,C) to (N,2H,W,C)
        img_repeated = torch.repeat_interleave(img[np.newaxis, ...], len(ref_imgs), dim=0)

        # SegGPT uses a weird format--it needs images/masks to be in format (N,2H,W,C)--where the first H rows are the reference image, and the next H rows are the input image.
        imgs = torch.cat((ref_imgs, img_repeated), dim=1)
        masks = torch.cat((ref_masks, ref_masks), dim=1)

        torch.manual_seed(2)
        output = self.run_one_image(imgs, masks, self.model, device)

        return output

    def loss_fn(self,pred,gt):
        # pred is (N,H,W,C)
        # gt is (N,H,W,C)
        return F.l1_loss(pred,gt)
    
    def get_optimizer(self,lr):
        # parameters are the reference img and reference mask
        parameters = [self.reference_img,self.reference_mask]

        optimizer = torch.optim.Adam(parameters,lr=lr)
        return optimizer
    
# training loop