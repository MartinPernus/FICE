import os
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

def loss_dict2string(loss_dict):
    string = ''
    string += f'total: {sum(loss_dict.values()).item():.2f}'
    for k, v in loss_dict.items():
        string += f', {k}: {v.item():.2f}'
    return string

def read_image(path, unsqueeze=True):
    path = os.path.expanduser(path)
    img = to_tensor(Image.open(path).convert('RGB'))
    if unsqueeze:
        img = img.unsqueeze(0)
    return img

def load_imgdir(path):
    files = Path(path).glob('*.jpg')
    imgs = [read_image(file) for file in files]
    imgs = torch.cat(imgs, dim=0)
    return imgs

def extract_from_statedict(statedict, name:str = 'model'):
    if name.endswith('.'):
        name = name[:-1]
    newdict = {k[len(name)+1:]: v for k, v in statedict.items() if k.startswith(name)}
    return newdict

def images2grid(*images, size=256, **grid_kwargs):
    """Creates a grid of images given a list of image arguments"""
    images = [img.cpu().view(-1, *img.shape[-3:]) for img in images]
    if size is not None:
        images = [F.interpolate(x, size=size, mode='bilinear', align_corners=False) for x in images]

    if not 'nrow' in grid_kwargs:
        grid_kwargs['nrow'] = len(images)
    
    images = torch.stack(images, dim=1)
    images = images.view(-1, *images.shape[-3:])

    grid = make_grid(images, **grid_kwargs)
    grid = grid.clamp(0, 1)
    return grid
