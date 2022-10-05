import argparse
import torch.nn.functional as F
import torch.nn as nn
import os
from typing import List
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

from .densepose import add_densepose_config

def setup_config(config_fpath: str, args: argparse.Namespace):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = 'checkpoints/densepose.pkl'
    cfg.freeze()
    return cfg

thisdir = os.path.dirname(__file__)
def set_instances(box, bs):
    box = Boxes(box)
    instances = [{'pred_classes': torch.tensor([0], device=box.device), 'pred_boxes': box} for _ in range(bs)]
    instances = [Instances(image_size=(256, 256), **x) for x in instances]
    return instances

class DenseNet(nn.Module):
    def __init__(self, indexes=None):
        super().__init__()
        args = torch.load(os.path.join(thisdir, 'args_dict.pt'))
        args = argparse.Namespace(**args)
        cfg = setup_config(os.path.join(thisdir, args.cfg), args)
        self.model = DefaultPredictor(cfg).model
        self.body_indexes = indexes or [1,2,3,4,15,16,17,18,19,20,21,22, 23, 24]

    def forward(self, img, instances):  # img in [0, 1]
        img = img[:, [2, 1, 0], :, :]   # rgb to bgr
        img = img * 255
        height, width = img.shape[-2:]
        dev = img.device

        inputs = [{'image': i_img, 'height': height , 'width': width} for i_img in img]

        output = self.model.inference(inputs, instances, do_postprocess=False)
        output = [x._fields for x in output]

        bbox = [x['pred_boxes'].tensor for x in output]
        poses = [x['pred_densepose'] for x in output]

        mask_bbox = torch.tensor([x.size(0) > 0 for x in bbox]).to(dev)
        bbox = torch.stack([x[0] if x.size(0) > 0 else torch.ones(4, device=dev) for x in bbox])

        coarse = [x.coarse_segm for x in poses]
        mask_seg = torch.tensor([len(x) > 0 for x in coarse]).to(dev)
        coarse = [x[0] if len(x) > 0 else torch.empty(2, 112, 112, device=dev).fill_(-10000.) for x in coarse]
        coarse = torch.stack(coarse)
        coarse = torch.softmax(coarse, dim=1)
        coarse = coarse[:, 1].unsqueeze(1)

        fine = [x.fine_segm for x in poses]
        fine = [x[0] if x.size(0) > 0 else torch.empty(25, 112, 112, device=img.device).fill_(-10000.) for x in fine]

        fine = torch.stack(fine)
        fine = torch.softmax(fine, dim=1)

        fine = fine * coarse  
        # mask = mask_seg & mask_bbox

        body = fine[:, self.body_indexes]
        
        return body
