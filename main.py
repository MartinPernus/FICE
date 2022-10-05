import os
import argparse

from easydict import EasyDict as edict
from torchvision.utils import save_image

from utils import *
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--description', type=str, default=None)
parser.add_argument('--input_dir', type=str, default=None)
args = parser.parse_args()

description = args.description or 'long sleeve silk crepe de chine shirt featuring graphic pattern printed in tones of blue'
input_dir = args.input_dir or 'imgs/input'

device = 'cuda:%d' % args.gpu
opt = edict(bs=args.batch_size)
trainer = Trainer(opt)
trainer.to(device)

imgs = load_imgdir(input_dir).to(device)

outdir = 'output'
os.makedirs(outdir, exist_ok=True)

def get_filename(sentence):
    sent_temp = sentence.replace('/', '-')  # avoid difficulties with creation of new folder
    filename = os.path.join(outdir, sent_temp + '.jpg')
    return filename

def save_comparison(imgs_final, filename):
    imgs_compare = torch.cat((imgs, imgs_final), dim=0).cpu()
    save_image(imgs_compare, filename, nrow=len(imgs))

print('Target description:', description)
filename = get_filename(description)

trainer.init(imgs, description)
imgs_final = trainer.process()
save_comparison(imgs_final, filename)
