# GaussianCube: Structuring Gaussian Splatting for 3D Generative Modeling using Optimal Transport

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Yiji Cheng](https://www.linkedin.com/in/yiji-cheng-a8b922213/?originalSubdomain=cn), [Jiaolong Yang](https://jlyang.org/), [Chunyu Wang](https://www.chunyuwang.org/), [Feng Zhao](https://en.auto.ustc.edu.cn/2021/0616/c26828a513169/page.htm), [Yansong Tang](https://andytang15.github.io/), [Dong Chen](http://www.dongchen.pro/), [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).

[Paper]() | [Project Page](https://gaussiancube.github.io/) | [Code](https://github.com/GaussianCube/GaussianCube)

## Abstract

> In this work, we introduce GaussianCube, a novel representation crafted for 3D generative modeling. Recent advances in Gaussian Splatting have demonstrated considerable improvements in expressiveness and efficiency. However, its unstructured nature poses a significant challenge to its deployment in generative applications. To fully unleash the potential of Gaussian Splatting in generative modeling, we present GaussianCube, a powerful and efficient spatial-structured representation for 3D generation. We propose to perform densification-constrained fitting to obtain fix-length Gaussians for each 3D asset, and then systematically arrange the Gaussians into the structured voxel grid via Optimal Transport. Furthermore, we perform generative modeling on the proposed GaussianCube using 3D diffusion models. The spatial coherence of GaussianCube allows us to utilize a standard 3D U-Net as our diffusion network without elaborate design. Extensive experiments conducted on ShapeNet and challenging large-vocabulary OmniObject3D show the superiority to prior works, underscoring its potential as a versatile 3D representation.

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

## Acknowledgement

This codebase is built upon the [improved-diffusion](https://github.com/openai/improved-diffusion), thanks to the authors for their great work.

## Todo

- [x] Release the inference code.
- [x] Release all pretrained models.
- [ ] Release the data fitting code.
- [ ] Release the diffusion training code.