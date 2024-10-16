# GaussianCube: A Structured and Explicit Radiance Representation for 3D Generative Modeling [NeurIPS 2024]

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Yiji Cheng](https://www.linkedin.com/in/yiji-cheng-a8b922213/?originalSubdomain=cn), [Jiaolong Yang](https://jlyang.org/), [Chunyu Wang](https://www.chunyuwang.org/), [Feng Zhao](https://en.auto.ustc.edu.cn/2021/0616/c26828a513169/page.htm), [Yansong Tang](https://andytang15.github.io/), [Dong Chen](http://www.dongchen.pro/), [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).

[Paper](https://arxiv.org/abs/2403.19655) | [Project Page](https://gaussiancube.github.io/) | [Code](https://github.com/GaussianCube/GaussianCube)


https://github.com/GaussianCube/GaussianCube/assets/164283176/3935590f-a36a-4bdc-b9e1-520c35b9f53e


## Abstract

> We introduce a radiance representation that is both structured and fully explicit and thus greatly facilitates 3D generative modeling. Existing radiance representations either require an implicit feature decoder, which significantly degrades the modeling power of the representation, or are spatially unstructured, making them difficult to integrate with mainstream 3D diffusion methods. We derive GaussianCube by first using a novel densification-constrained Gaussian fitting algorithm, which yields high-accuracy fitting using a fixed number of free Gaussians, and then rearranging these Gaussians into a predefined voxel grid via Optimal Transport. Since GaussianCube is a structured grid representation, it allows us to use standard 3D U-Net as our backbone in diffusion modeling without elaborate designs. More importantly, the high-accuracy fitting of the Gaussians allows us to achieve a high-quality representation with orders of magnitude fewer parameters than previous structured representations for comparable quality, ranging from one to two orders of magnitude. The compactness of GaussianCube greatly eases the difficulty of 3D generative modeling. Extensive experiments conducted on unconditional and class-conditioned object generation, digital avatar creation, and text-to-3D synthesis all show that our model achieves state-of-the-art generation results both qualitatively and quantitatively, underscoring the potential of GaussianCube as a highly accurate and versatile radiance representation for 3D generative modeling.

## Environment Setup

We recommend Linux for performance and compatibility reasons. We use conda to manage the environment. Please install conda from [here](https://docs.conda.io/en/latest/miniconda.html) if you haven't done so.

```
git clone https://github.com/GaussianCube/GaussianCube.git
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
python inference.py --exp_name /tmp/shapenet_car_test --config configs/shapenet_uncond.yml  --rescale_timesteps 300 --ckpt ./shapenet_car/shapenet_car_ckpt.pt  --mean_file ./shapenet_car/mean.pt  --std_file ./shapenet_car/std.pt  --bound 0.45 --num_samples 10 --render_video
```

To inference pretrained model of ShapeNet Chair, save the downloaded model checkpoint and dataset statistics to `./shapenet_chair/`, then run:
```bash
python inference.py --exp_name /tmp/shapenet_chair_test --config configs/shapenet_uncond.yml  --rescale_timesteps 300 --ckpt ./shapenet_chair/shapenet_chair_ckpt.pt  --mean_file ./shapenet_chair/mean.pt  --std_file ./shapenet_chair/std.pt  --bound 0.35 --num_samples 10 --render_video
```

### Mesh Conversion

For the generated results, we provide a script to convert the generated GaussianCube to mesh following [LGM](https://github.com/3DTopia/LGM). First, install additional dependencies:

```bash
# for mesh extraction
pip install nerfacc
pip install git+https://github.com/NVlabs/nvdiffrast
# install diff_gauss for alpha rendering
git clone --recurse-submodules https://github.com/slothfulxtx/diff-gaussian-rasterization.git 
cd diff-gaussian-rasterization
python setup.py install
```

Then run the following command to convert the generated results to mesh:
```bash
python scripts/convert_mesh.py --test_path /tmp/shapenet_car_test/rank_00_0000.pt --cam_radius 1.2 --bound 0.45 --mean_file ./shapenet_car/mean.pt --std_file ./shapenet_car/std.pt
```

## Training

### Data Preparation

Please refer to [data_construction](https://github.com/GaussianCube/GaussianCube_Construction) to prepare the training data. Then, put the data in the following structure (take ShapeNet as an example):

```
example_data
├── shapenet
│   ├── mean_volume_act.pt
│   ├── std_volume_act.pt
│   ├── shapenet_train.txt
│   ├── volume_act/
│   │   ├── gs_cube_0000.pt
│   │   ├── gs_cube_0001.pt
│   │   └── ...
│   └── shapenet_rendering_512
```

The `mean_volume_act.pt` and `std_volume_act.pt` are the pre-computed mean and std files of the training data in `volume_act/`. The `volume_act` folder contains the pre-processed GaussianCube data. The `shapenet_rendering_512` folder contains the rendered images of the training data. The `shapenet_train.txt` is the text file containing the list of training data, like:

```
gs_cube_0000
gs_cube_0001
...
```

### Unconditional Diffusion Training on ShapeNet Car or ShapeNet Chair

Run the following command to train the model:
```bash
mpiexec -n 8 python main.py --log_interval 100 --batch_size 8 --lr 5e-5 --exp_name ./output/shapenet_diffusion_training --save_interval 5000 --config configs/shapenet_uncond.yml --use_tensorboard --use_vgg --load_camera 1 --render_l1_weight 10 --render_lpips_weight 10 --use_fp16 --mean_file ./example_data/shapenet/mean_volume_act.pt --std_file ./example_data/shapenet/std_volume_act.pt --data_dir ./example_data/shapenet/volume_act --cam_root_path ./example_data/shapenet/shapenet_rendering_512/ --txt_file ./example_data/shapenet/shapenet_train.txt --bound 0.45 --start_idx 0 --end_idx 100 --clip_input
```

### Class-conditioned Diffusion Training on OmniObject3D

Run the following command to train the model:
```bash
mpiexec -n 8 python main.py --log_interval 100 --batch_size 8 --lr 5e-5 --exp_name ./output/omniobject3d_diffusion_training --save_interval 5000 --config configs/omni_class_cond.yml --use_tensorboard --use_vgg --load_camera 1 --render_l1_weight 10 --render_lpips_weight 10 --use_fp16 --mean_file ./example_data/omniobject3d/mean_volume_act.pt --std_file ./example_data/omniobject3d/std_volume_act.pt --data_dir ./example_data/omniobject3d/volume_act --cam_root_path ./example_data/omniobject3d/Omniobject3d_rendering_512/ --txt_file ./example_data/omniobject3d/omni_train.txt --uncond_p 0.2 --bound 1.0 --start_idx 0 --end_idx 100 --clip_input --omni
```

### Text-conditioned Diffusion Training on Objaverse

Extract the CLIP features of text captions and put them under `./example_data/objaverse/` using the following script:
```bash
python scripts/encode_text_feature.py
```

Then run the following command to train the model:
```bash
mpiexec -n 8 python main.py --log_interval 100 --batch_size 8 --lr 5e-5 --weight_decay 0 --exp_name ./output/objaverse_diffusion_training --save_interval 5000 --config configs/objaverse_text_cond.yml --use_tensorboard --use_vgg --load_camera 1 --render_l1_weight 10 --render_lpips_weight 10 --use_fp16 --data_dir ./example_data/objaverse/volume_act/ --start_idx 0 --end_idx 100 --txt_file ./example_data/objaverse/objaverse_train.txt --mean_file ./example_data/objaverse/mean_volume_act.pt --std_file ./example_data/objaverse/std_volume_act.pt --cam_root_path ./example_data/objaverse/objaverse_rendering_512/ --bound 0.5 --uncond_p 0.2 --objaverse --clip_input --text_feature_root ./example_data/objaverse/objaverse_text_feature/
```

### Image-conditioned Diffusion Training on Synthetic Avatar

Extract the DINO features of avatars and put them under `./example_data/avatar/` using the following script:
```bash
python scripts/encode_dino_feature.py
```

Then run the following command to train the model:
```bash
python main.py --log_interval 100 --batch_size 8 --lr 5e-5 --weight_decay 0 --exp_name ./output/avatar_diffusion_training --save_interval 5000 --config configs/avatar_img_cond.yml --use_tensorboard --use_vgg --load_camera 1 --render_l1_weight 10 --render_lpips_weight 10 --use_fp16 --data_dir ./example_data/avatar/volume_act/ --start_idx 0 --end_idx 100 --txt_file ./example_data/avatar/avatar_train.txt --mean_file ./example_data/avatar/mean_volume_act.pt --std_file ./example_data/avatar/std_volume_act.pt --cam_root_path ./example_data/avatar/avatar_rendering_512/ --bound 0.5 --uncond_p 0.2 --avatar --clip_input --text_feature_root ./example_data/avatar/avatar_dino_feature/
```

## Acknowledgement

This codebase is built upon the [improved-diffusion](https://github.com/openai/improved-diffusion), thanks to the authors for their great work. Also thanks the authors of [Cap3D](https://arxiv.org/abs/2306.07279) and [VolumeDiffusion](https://arxiv.org/abs/2312.11459) for the text captions of Objaverse dataset.

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
- [x] Release the diffusion training code.
