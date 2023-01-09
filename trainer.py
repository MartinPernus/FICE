from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from utils import images2grid
from models.clip.model import CLIP
from models.stylegan.model import load_gan
from models.e4e.model import load_e4e
from models.densepose.model import DenseNet, set_instances
from models.segmentation.model import SegModel

def predict_w(img):
    e4e = load_e4e()
    e4e = e4e.to(img.device)
    w = e4e(img)
    return w

def img_loss_fn(im1, im2):
    loss = F.mse_loss(im1, im2, reduction='none')
    loss = loss.mean((1,2,3)).sum()
    return loss

def shape_loss_fn(im1, im2):
    loss = F.mse_loss(im1, im2, reduction='none')
    loss = loss.mean((1,2,3)).sum()
    return loss

def clip_loss_fn(clip_similarity):
    loss = (1 - clip_similarity).sum()
    return loss

def w_delta_loss_fn(w):
    N = w.size(1)
    w_ref = w[:, 0].unsqueeze(1).repeat(1, N-1, 1)
    w_tar = w[:, 1:]
    loss = F.mse_loss(w_ref, w_tar, reduction='none')
    loss = loss.mean((1,2)).sum()
    return loss

def resize(x):
    return F.interpolate(x, (256, 256), mode='bilinear')

class Model(nn.Module):
    def __init__(self, clip_model) -> None:
        super().__init__()
        self.gan = load_gan()
        self.clip = CLIP(model_name=clip_model)

        self.densenet = DenseNet()
        self.segnet = SegModel()

        w_mean, w_std = self.gan.get_latent_stats()
        self.register_buffer('w_mean', w_mean)
        self.register_buffer('w_std', w_std)

    def clip_similarity(self, imgs, text_feats):
        img_features = self.clip.encode_image(imgs)
        clip_similarity = self.clip.compute_similarity(img_features, text_feats)
        return clip_similarity

    def densenet_forward(self, imgs):
        bs = imgs.size(0)
        box = torch.tensor([[0, 0, 256, 256]], device=imgs.device)
        instances = set_instances(box, bs)
        body = self.densenet(imgs, instances)
        return body

    def deeplab_seg_head(self, imgs):
        _, _, head = self.segnet(imgs)
        return head

class Trainer():
    def __init__(self, opt={}):
        self.opt = self.set_defaults(opt)
        self.model = Model(clip_model=opt.clip_model).eval().requires_grad_(False)

    def set_defaults(self, opt):
        weights_dict = {'clip': 1,
                        'img': 30,
                        'shape': 10,
                        'head_shape': 1,
                        'w_delta': 1}

        opt_default = dict(weights_dict=weights_dict,
                           lr=5e-2,
                           n_iters=500,
                           clip_model='RN50x4')

        for key in opt_default:
            if key not in opt:
                opt[key] = opt_default[key]
        return opt

    def to(self, device):
        self.model.to(device)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def init(self, imgs_batch, sentence, img_save_dir=None):
        self.imgs_batch = imgs_batch.to(self.device)

        with torch.no_grad():
            text_feats = self.model.clip.encode_text(sentence).view(1, -1)   # check the dimensionality
            shape_real = self.model.densenet_forward(imgs_batch)
            _, body_mask, head_mask = self.model.segnet(imgs_batch)

        self.shape_real = shape_real
        self.img_mask = 1 - body_mask

        self.blend_mask = 1 - head_mask
        self.text_feats = text_feats
        self.head_shape_init = head_mask

        w = predict_w(imgs_batch).cpu()
        self.w = w.clone().to(self.device).requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.w], lr=self.opt.lr)

        if img_save_dir is not None:
            img_save_dir = Path(img_save_dir)
            save_image(self.img_mask, img_save_dir/'img_mask.jpg')
            save_image(self.blend_mask, img_save_dir/'blend_mask.jpg')


    def forward(self, n_log=None):
        pbar = tqdm(range(self.opt.n_iters+1))

        for i in pbar:
            w = self.w

            imgs_gen = self.model.gan.gen_w(w)
            body_shape = self.model.densenet_forward(imgs_gen)
            head_shape = self.model.deeplab_seg_head(imgs_gen)

            loss_dict = {}

            loss_dict['img'] = img_loss_fn(self.img_mask*imgs_gen, self.img_mask*self.imgs_batch)
            loss_dict['shape'] = shape_loss_fn(self.shape_real, body_shape)
            loss_dict['head_shape'] = shape_loss_fn(self.head_shape_init, head_shape)
            loss_dict['w_delta'] = w_delta_loss_fn(w)

            clip_sim = self.model.clip_similarity(imgs_gen, self.text_feats)
            loss_dict['clip'] =  clip_loss_fn(clip_sim)

            loss_dict_scaled = {k: loss_dict[k]*self.opt.weights_dict[k] for k in loss_dict}
            loss = sum(loss_dict_scaled.values())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            pbar.set_description(f'{loss.item():.3f}')

            if n_log is not None and i % n_log == 0:
                ##### image composition
                with torch.no_grad():
                    imgs_final = self.blend_mask * imgs_gen + (1-self.blend_mask) * self.imgs_batch

                yield imgs_final, imgs_gen, loss_dict_scaled

        with torch.no_grad():
            imgs_final = self.blend_mask * imgs_gen + (1-self.blend_mask) * self.imgs_batch
        yield imgs_final, imgs_gen, loss_dict_scaled

    def process_gen(self, n_log=20):
        for imgs_final, imgs_gen, loss_dict in self.forward(n_log=n_log):
            grid = images2grid(self.imgs_batch, imgs_gen, imgs_final)
            yield grid, loss_dict

    def process(self):
        for imgs_final, _, _ in self.forward(n_log=None):
            pass
        return imgs_final


