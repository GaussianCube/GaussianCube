# GaussianCube: A Structured and Explicit Radiance Representation for 3D Generative Modeling

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Yiji Cheng](https://www.linkedin.com/in/yiji-cheng-a8b922213/?originalSubdomain=cn), [Jiaolong Yang](https://jlyang.org/), [Chunyu Wang](https://www.chunyuwang.org/), [Feng Zhao](https://en.auto.ustc.edu.cn/2021/0616/c26828a513169/page.htm), [Yansong Tang](https://andytang15.github.io/), [Dong Chen](http://www.dongchen.pro/), [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).

[Paper](https://arxiv.org/abs/2403.19655) | [Project Page](https://gaussiancube.github.io/) | [Code](https://github.com/GaussianCube/GaussianCube)

## Abstract

> We introduce a radiance representation that is both structured and fully explicit and thus greatly facilitates 3D generative modeling. Existing radiance representations either require an implicit feature decoder, which significantly degrades the modeling power of the representation, or are spatially unstructured, making them difficult to integrate with mainstream 3D diffusion methods. We derive GaussianCube by first using a novel densification-constrained Gaussian fitting algorithm, which yields high-accuracy fitting using a fixed number of free Gaussians, and then rearranging these Gaussians into a predefined voxel grid via Optimal Transport. Since GaussianCube is a structured grid representation, it allows us to use standard 3D U-Net as our backbone in diffusion modeling without elaborate designs. More importantly, the high-accuracy fitting of the Gaussians allows us to achieve a high-quality representation with orders of magnitude fewer parameters than previous structured representations for comparable quality, ranging from one to two orders of magnitude. The compactness of GaussianCube greatly eases the difficulty of 3D generative modeling. Extensive experiments conducted on unconditional and class-conditioned object generation, digital avatar creation, and text-to-3D synthesis all show that our model achieves state-of-the-art generation results both qualitatively and quantitatively, underscoring the potential of GaussianCube as a highly accurate and versatile radiance representation for 3D generative modeling.

## Environment Setup

We recommend Linux for performance and compatibility reasons. We use conda to manage the environment. Please install conda from [here](https://docs.conda.io/en/latest/miniconda.html) if you haven't done so.

```
git clone https://github.com/GaussianCube
cd GaussianCube
conda env create -f environment.yml
conda activate gaussiancube
```

## Model Download

Please download model checkpoints and dataset statistics (pre-computed mean and sta files) from the following links:

### Huggingface

| Model                 | Task                          | Download                                                                          |
|-----------------------|-------------------------------|-----------------------------------------------------------------------------------|
| OmniObject3D          | Class-conditioned Generation  | [🤗 Hugging Face](https://huggingface.co/BwZhang/GaussianCube-OmniObject3D-v1.0)  |
| ShapeNet Car          | Unconditional Generation      | [🤗 Hugging Face](https://huggingface.co/BwZhang/GaussianCube-ShapeNetCar-v1.0)   |
| ShapeNet Chair        | Unconditional Generation      | [🤗 Hugging Face](https://huggingface.co/BwZhang/GaussianCube-ShapeNetChair-v1.0) |

## Inference

### Class-conditioned Generation on OmniObject3D

To inference pretrained model of OmniObject3D, save the downloaded model checkpoint and dataset statistics to `./OmniObject3D/`, then run:
```bash
python inference.py --exp_name /tmp/OmniObject3D_test --config configs/omni_class_cond.yml  --rescale_timesteps 300 --ckpt ./OmniObject3D/OmniObject3D_ckpt.pt  --mean_file ./OmniObject3D/mean.pt --std_file ./OmniObject3D/std.pt  --bound 1.0 --num_samples 10 --render_video --class_cond
```

### Unconditional Generation on ShapeNet

To inference pretrained model of ShapeNet Car, save the downloaded model checkpoint and dataset statistics to `./shapenet_car/`, then run:
```bash
python render_inference.py --exp_name /tmp/shapenet_car_test --config configs/shapenet_uncond.yml  --rescale_timesteps 300 --ckpt ./shapenet_car/shapenet_car_ckpt.pt  --mean_file ./shapenet_car/mean.pt  --std_file ./shapenet_car/std.pt  --bound 0.45 --num_samples 10 --render_video
```

To inference pretrained model of ShapeNet Chair, save the downloaded model checkpoint and dataset statistics to `./shapenet_chair/`, then run:
```bash
python inference.py --exp_name /tmp/shapenet_chair_test --config configs/shapenet_uncond.yml  --rescale_timesteps 300 --ckpt ./shapenet_chair/shapenet_chair_ckpt.pt  --mean_file ./shapenet_chair/mean.pt  --std_file ./shapenet_chair/std.pt  --bound 0.35 --num_samples 10 --render_video
```

## Training

### Data Preparation

Please refer to [data_construction](https://github.com/GaussianCube/GaussianCube_Construction).

## Acknowledgement

This codebase is built upon the [improved-diffusion](https://github.com/openai/improved-diffusion), thanks to the authors for their great work.

## Citation

If you find this work useful, please consider citing:
```
@article{zhang2024gaussiancube,
  title={GaussianCube: Structuring Gaussian Splatting using Optimal Transport for 3D Generative Modeling},
  author={Zhang, Bowen and Cheng, Yiji and Yang, Jiaolong and Wang, Chunyu and Zhao, Feng and Tang, Yansong and Chen, Dong and Guo, Baining},
  journal={arXiv preprint arXiv:2403.19655},
  year={2024}
}
```

## Todo

- [x] Release the inference code.
- [x] Release all pretrained models.
- [x] Release the data construction code.
- [ ] Release the diffusion training code.