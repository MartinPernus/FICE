# FICE: Text-Conditioned Fashion Image Editing with Guided GAN Inversion ([arXiv](http://arxiv.org/abs/2301.02110))

<img src=imgs/paper/example.png width="1000">


## Installation
`cat requirements.txt | xargs -n 1 -L 1 pip install`

## Download Models
`./download.sh`

## Example Usage (Inference)
`python main.py --input_dir imgs/input --description "long sleeve silk crepe de chine shirt featuring graphic pattern printed in tones of blue"`

The `--input_dir` argument specifies directory of images (256x256 resolution) to be edited.

## (New) Intructions on Training With Other Datasets
1. Train the GAN model using the [StyleGAN2 repository](https://github.com/NVlabs/stylegan2-ada-pytorch)
2. Convert the best .pkl file (lowest FID score) to .pt file with provided script in `scripts/pkl2pt` directory. The `main.py` in this directory has to be run from this directory! You can simply place a .pkl file in the `target` directory and the result will be placed in the `result` directory.
3. Run the E4e training from `misc_scripts/E4e` directory. This is only a slight modification of the original [E4e](https://github.com/omertov/encoder4editing) repository, where most edits happen in `models/psp.py` file to enable the proper GAN code. Make sure to edit the `scripts/train.py` file with your custom arguments.
4. (optional) Depending on the dataset and your purpose you might need to train a segmentation model that supports lower body regions as well. The training procedure follows common segmentation training regimes and should be easy to perform. Nevertheless, finding a good dataset for such segmentation training could be a problem!


## Code Acknowledgements
[Encoder for Editing](https://github.com/omertov/encoder4editing) 

[DensePose](https://github.com/facebookresearch/DensePose) 

[StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) 

[CLIP](https://github.com/openai/CLIP)

## Sponsor Acknowledgements
Supported in parts by the Slovenian Research Agency ARRS through the Research Programme P2-0250(B) Metrology and Biometric System, the ARRS Project J2-2501(A) DeepBeauty and the ARRS junior researcher program.

<img src=imgs/ARRSLogo.png width="400">

