import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from mpi4py import MPI

from model.unet import UNetModel
from model.dpmsolver import NoiseScheduleVP, model_wrapper, DPM_Solver
from utils import dist_util, logger
from utils.script_util import create_gaussian_diffusion, init_volume_grid, build_single_viewpoint_cam
from dataset.dataset_render import load_data
from gaussian_renderer import render
import imageio
from tqdm import tqdm
import glob


MODEL_TYPES = {
    'xstart': 'x_start',
    'v': 'v',
    'eps': 'noise',
}


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    print("Start inference...")
    args = create_argparser().parse_args()

    model_and_diffusion_config = OmegaConf.load(args.config)
    print("Model and Diffusion config: ", model_and_diffusion_config)

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())
    seed_everything(args.seed + dist.get_rank())

    model_and_diffusion_config['model']['precision'] = "32"
    model = UNetModel(**model_and_diffusion_config['model'])

    diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        print("Loaded ckpt: ", args.ckpt)

    logger.configure(args.exp_name)
    options = logger.args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    model.to(dist_util.dev())
    model.eval()
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    val_data = load_data(
        batch_size=1,
        deterministic=True,
        class_cond=args.class_cond,
    )

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(dist_util.dev()))
    std_volume = torch.tensor(init_volume_grid(bound=args.bound, num_pts_each_axis=32)).to(torch.float32).to(dist_util.dev()).contiguous()
    bg_color = torch.tensor([1,1,1]).to(torch.float32).to(dist_util.dev())
    mean = torch.load(args.mean_file).to(torch.float32).to(dist_util.dev())
    std = torch.load(args.std_file).to(torch.float32).to(dist_util.dev())

    mean = mean.permute(3, 0, 1, 2).requires_grad_(False).contiguous()
    std = std.permute(3, 0, 1, 2).requires_grad_(False).contiguous()

    img_id = 0
    val_psnrs = []
    num_batch_per_rank = args.num_samples // dist.get_world_size()
    for _ in tqdm(range(num_batch_per_rank)):
        
        model_kwargs = next(val_data)  

        image_size = model_and_diffusion_config['model']['image_size']
        sample_shape = (1, model_and_diffusion_config['model']['in_channels'], image_size, image_size, image_size)

        condition, unconditional_condition = {}, {}
        if args.class_cond:
            condition['class_labels'] = model_kwargs['class_labels'].to(dist_util.dev())
            unconditional_condition['class_labels'] = torch.zeros_like(model_kwargs['class_labels']).to(dist_util.dev()) + 216

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type=MODEL_TYPES[model_and_diffusion_config['diffusion']['predict_type']],
            model_kwargs={},
            guidance_type='uncond' if not args.class_cond else 'classifier-free',
            guidance_scale=args.guidance_scale,
            condition=None if not args.class_cond else condition,
            unconditional_condition=None if not args.class_cond else unconditional_condition,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')

        with torch.no_grad():
            noise = torch.randn(sample_shape, device=dist_util.dev()) * args.temperature

            samples = dpm_solver.sample(
                x=noise,
                steps=args.rescale_timesteps,
                t_start=1.0,
                t_end=1/1000,
                order=3 if not args.class_cond else 2,
                skip_type='time_uniform',
                method='multistep',
            )
            samples_denorm = samples * std + mean

            frames = []
            for i, cam_info in enumerate(model_kwargs["cams"]):
                cam = build_single_viewpoint_cam(cam_info, 0)
                res = render(cam, samples_denorm[0], std_volume, bg_color, args.active_sh_degree)
                
                s_path = os.path.join(logger.get_dir(), 'render_images')
                os.makedirs(s_path,exist_ok=True)
                output_image = res["render"].clamp(0.0, 1.0)

                rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
                rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                imageio.imwrite(os.path.join(s_path, "rank_{:02}_render_{:06}_cam_{:02}.png".format(dist.get_rank(), img_id, i)), rgb_map)

                frames.append(rgb_map)
        
            if args.render_video:
                s_path = os.path.join(logger.get_dir(), 'videos')
                os.makedirs(s_path,exist_ok=True)
                imageio.mimwrite(os.path.join(s_path, "rank_{:02}_render_{:06}.mp4".format(dist.get_rank(), img_id)), frames, fps=30)

        img_id += 1

 
def create_argparser():
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # Model config
    parser.add_argument("--config", type=str, default="configs/shapenet_uncond.yml")
    # Data args
    parser.add_argument("--mean_file", type=str, default="./shapenet_car/mean.pt")
    parser.add_argument("--std_file", type=str, default="./shapenet_car/std.pt")
    parser.add_argument("--active_sh_degree", type=int, default=0)
    parser.add_argument("--bound", type=float, default=0.45)
    # Inference args
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rescale_timesteps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--class_cond", action="store_true")
    parser.add_argument("--render_video", action="store_true")
 
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
