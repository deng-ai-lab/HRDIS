# Hybrid Regularization Improves Diffusion-based Inverse Problem Solving (ICLR 2025) [Paper](https://openreview.net/pdf?id=d7pr2doXn3)

This is the README file for the implementaiton of *Hybrid Regularization Empowers Diffusion-based Inverse Problem Solving (HRDIS)*. 

<p align="center">
<img src="fig/intro.png" width="1000">
<br>
<em> 🌟Comparison between <a href="https://arxiv.org/abs/2305.04391">RED-diff</a> and our proposed HRDIS. </em>
</p>

## Installation

Download [ImageNet](https://image-net.org/) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. You need to write your data directory at _configs\dataset\imagenet256_val(ffhq256_val).yaml.

```
name: "ImageNet_256x256"
root: "./data/imagenet"
split: "val"
image_size: 256
channels: 3
meta_root: "_exp"
transform: "diffusion"
subset_txt: "misc/dgp_top1k.txt"  
```

Download pretrained checkpoints and put them in _exp/ckpts as following file.

| Dataset                                         | File                                      | Model Source |
|-----------------------------------------------------|-------------------------------------------|-------------------------------------------|
| ImageNet  | imagenet/256x256_diffusion_uncond.pt      |[guided-diffusion](https://github.com/openai/guided-diffusion)|
|FFHQ      | ffhq/ffhq_10m.pt  |[DPS](https://github.com/DPS2022/diffusion-posterior-sampling)|


Install the dependencies: 

```
pip install -r requirements.txt
```
Git clone external codes for non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
```

Download mask for inapinting from the [Palette](https://arxiv.org/abs/2111.05826) and put it in _exp/masks/20ff.npz

## Inference

Select the test dataset by enabling the following code in ./main.py

```
@hydra.main(version_base="1.2", config_path="_configs", config_name="imagenet256_uncond")
#@hydra.main(version_base="1.2", config_path="_configs", config_name="ffhq256_uncond")
```

Tune the hyperparamaters interactively using sampling script: 

```
sh sample_test.sh 
```

## Reference

```
@inproceedings{
dou2025hybrid,
title={Hybrid Regularization Improves Diffusion-based Inverse Problem Solving},
author={Hongkun Dou and Zeyu Li and Jinyang Du and Lijun Yang and Wen Yao and Yue Deng},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=d7pr2doXn3}
}
```
