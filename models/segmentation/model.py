import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50

class SegModel(nn.Module):  # todo: move to models
    def __init__(self) -> None:
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=False, num_classes=3, pretrained_backbone=True)
        ckpt = 'checkpoints/segmentator.pt'
        ckpt = torch.load(ckpt, map_location='cpu')['state']
        ckpt = {k: v for k, v in ckpt.items() if k != 'loss.weight'}
        self.load_state_dict(ckpt)
        self.eval().requires_grad_(False)

    def forward(self, x):
        x = self.model(x)['out']
        x = F.softmax(x, dim=1)
        
        background = x[:,0].unsqueeze(1)
        body = x[:,1].unsqueeze(1)
        head = x[:,2].unsqueeze(1)
        return background, body, head