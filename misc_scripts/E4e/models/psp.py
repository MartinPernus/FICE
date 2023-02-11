import torch
from torch import nn
from models.encoders import psp_encoders
from .StyleGAN import Generator
from pathlib import Path

from configs.paths_config import model_paths


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()

        self.decoder = self.set_decoder(ckpt_file=opts.decoder_checkpoint)

        with torch.no_grad():
            z = torch.randn(10000, 512)
            w = self.decoder.mapping(z)
            self.register_buffer('latent_avg', w.mean(0, keepdim=True))

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def set_decoder(self, ckpt_file):
        ckpt = torch.load(ckpt_file, map_location='cpu')
        decoder = Generator(to_01=False, **ckpt['G_kwargs'])
        configure_gan_noise(decoder, zero_out_noise=False)
        decoder.load_state_dict(ckpt['G'])
        decoder.eval()
        decoder.requires_grad_(False)
        return decoder


    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def forward(self, x, return_latent: bool, latent_mask=None, input_code=False,
                inject_latent=None, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images = self.decoder.synthesis(codes,
                                        input_type='w',
                                        noise_mode='random',
                                        force_fp32=True)

        if return_latent:
            return images, codes
        else:
            return images


def configure_gan_noise(G, zero_out_noise=True, noises=None):
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    if zero_out_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.zeros_like(buf)
    elif not zero_out_noise and noises is None:
        for key in noise_bufs:
            noise_bufs[key].copy_(torch.randn_like(noise_bufs[key]))
    elif not zero_out_noise and noises is not None:
        load_noises(G, noises)
    else:
        raise ValueError('When passing a value to noises, the "zero_out_noise" flag must be set to False!')


def load_noises(G, noises):
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    for key in noise_bufs:
        noise_bufs[key].copy_(noises[key])