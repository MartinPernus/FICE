import pickle
import glob
import os
from pathlib import Path
import torch
import dnnlib
from Models import Generator
from torchvision.utils import save_image

def to_vanilla_dict(in_dict):
    new_dict = {}
    for key, value in in_dict.items():
        if type(value) is dnnlib.util.EasyDict:
            new_dict[key] = to_vanilla_dict(value)
        else:
            new_dict[key] = value
    return new_dict

@torch.no_grad()
def pkl_to_dicts(pkl_file=None, savename=None, outdir=None):


    if outdir is None:
        outdir = Path(__file__).parent / 'StyleGAN_files'


    gen_state_file = outdir/f'{savename}-gen_statedict.pt'
    dis_state_file = outdir/f'{savename}-dis_statedict.pt'
    gen_kwargs_file = outdir/f'{savename}-gen_kwargs.pt'
    dis_kwargs_file = outdir/f'{savename}-dis_kwargs.pt'

    if all([os.path.exists(x) for x in (gen_state_file, dis_state_file, gen_kwargs_file, dis_kwargs_file)]):
        return
    else:
        with open(pkl_file, 'rb') as f:
            net = pickle.load(f)

        g_ema_kwargs = to_vanilla_dict(net['G_ema'].init_kwargs)
        dis_kwargs = to_vanilla_dict(net['D'].init_kwargs)

        assert len(net['G_ema'].init_args) == len(net['D'].init_args) ==  0

        torch.save(net['G_ema'].state_dict(), gen_state_file)
        torch.save(g_ema_kwargs, gen_kwargs_file)
        torch.save(net['D'].state_dict(), dis_state_file)
        torch.save(dis_kwargs, dis_kwargs_file)


def pkl_to_single_dict(infile, outfile=None):
    if type(infile) is str:
        infile = Path(infile)

    if outfile is None:
        outfile = Path('result', infile.stem + '.pt')

    if os.path.exists(outfile):
        print('The file already exists')
        return
    else:
        with open(infile, 'rb') as f:
            net = pickle.load(f)

        g_kwargs = to_vanilla_dict(net['G_ema'].init_kwargs)
        d_kwargs = to_vanilla_dict(net['D'].init_kwargs)

        outdict = {'G': net['G_ema'].state_dict(),
                   'D': net['D'].state_dict(),
                   'G_kwargs': g_kwargs,
                   'D_kwargs': d_kwargs}
        torch.save(outdict, outfile)



def test_singledict(infile):
    sd = torch.load(infile)
    gan_kwargs = sd['G_kwargs']

    G = Generator(**gan_kwargs).cuda()
    G.load_state_dict(sd['G'])
    z = torch.randn(4, 512).cuda()
    imgs = G(z)
    save_image(imgs, 'example_image.jpg', value_range=(-1, 1), normalize=True)
    print('done')


def main():
    pklfile_input = 'target/viton.pkl'
    outfile = 'result/viton.pt'
    pkl_to_single_dict(infile=pklfile_input, outfile=outfile)
    test_singledict(infile=outfile)

if __name__ == '__main__':
    main()