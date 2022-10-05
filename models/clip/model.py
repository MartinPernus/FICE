import torch.nn.functional as F
import torch.nn as nn
import torch
import clip
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def downsample(image, size):
    return F.interpolate(image, 2 * [size], mode='area')

def downsample_and_crop(image, size):
    image = F.interpolate(image, size, mode='bilinear')
    image = TF.center_crop(image, size)
    return image

class CLIP(nn.Module):
    def __init__(self, model_name='RN50x4'):
        assert model_name in ('RN50', 'RN101', 'RN50x4', 'ViT-B/32')
        super().__init__()
        self.clip_model, _ = clip.load(model_name, jit=False, device='cpu')
        self.clip_resolution = self.clip_model.visual.input_resolution
        self.embed_dim = self.clip_model.visual.output_dim

        self.register_buffer('clip_mean', torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1,3,1,1))
        self.register_buffer('clip_std', torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1,3,1,1))
        self.clip_normalize = T.Normalize(self.clip_mean, self.clip_std)

        self.clip_model.eval().requires_grad_(False)

    def _unnormalize(self, image):
        image = image * self.clip_std + self.clip_mean
        image = image.clamp(0, 1)
        return image

    def forward(self, image_01, text):
        image_features = self.encode_image(image_01)
        if type(text) is str:
            tokens = self.tokenize(text)
        else:
            tokens = text

        tokens = tokens.to(self.device)
        text_features = self.encode_tokens(tokens)
        similarity = self.compute_similarity(image_features, text_features)
        return similarity
    
    def compute_similarity(self, image_features, text_features):
        return image_features @ text_features.T

    def encode_image(self, image_01, img_is_square=True):
        if image_01.size(-1) > self.clip_resolution and img_is_square:
            mode = 'area'
        else:
            mode = 'bilinear'
        image_01 = F.interpolate(image_01, 2 * [self.clip_model.visual.input_resolution], mode=mode)

        image = self.clip_normalize(image_01)
        image_features = self.clip_model.encode_image(image)
        image_features = norm(image_features)
        return image_features
    
    def encode_tokens(self, tokens):
        text_features = self.clip_model.encode_text(tokens).detach()
        text_features = norm(text_features)
        return text_features

    def tokenize(self, text):
        return clip.tokenize(text, truncate=True)

    def encode_text(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        text_features = self.encode_tokens(tokens)
        return text_features

    @property
    def device(self):
        return next(self.parameters()).device


def norm(input):
    return input / input.norm(dim=-1, keepdim=True)
