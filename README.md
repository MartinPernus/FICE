# FICE: Text-Conditioned Fashion Image Editing with Guided GAN Inversion ([arXiv](http://arxiv.org/abs/2301.02110))

<img src=imgs/paper/example.png width="1000">


## Installation
`cat requirements.txt | xargs -n 1 -L 1 pip install`

## Download Models
`./download.sh`

## Example Usage
`python main.py --input_dir imgs/input --description "long sleeve silk crepe de chine shirt featuring graphic pattern printed in tones of blue"`

The `--input_dir` argument specifies directory of images (256x256 resolution) to be edited.

## Code Acknowledgements
[Encoder for Editing](https://github.com/omertov/encoder4editing) 

[DensePose](https://github.com/facebookresearch/DensePose) 

[StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) 

[CLIP](https://github.com/openai/CLIP)

## Sponsor Acknowledgements
Supported in parts by the Slovenian Research Agency ARRS through the Research Programme P2-0250(B) Metrology and Biometric System, the ARRS Project J2-2501(A) DeepBeauty and the ARRS junior researcher program.

<img src=imgs/ARRSLogo.png width="400">

