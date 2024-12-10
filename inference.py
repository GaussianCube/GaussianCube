import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from mpi4py import MPI
from huggingface_hub import hf_hub_download

from model.unet import UNetModel
from model.clip import FrozenCLIPEmbedder
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

# Model repository mapping
MODEL_REPOS = {
    "objaverse_v1.0": {
        "repo_id": "BwZhang/GaussianCube-Objaverse",
        "revision": "main",
        "model_path": "v1.0/objaverse_ckpt.pt",
        "mean_path": "v1.0/mean.pt",
        "std_path": "v1.0/std.pt",
        "bound": 0.5
    },
    "objaverse_v1.1": {
        "repo_id": "BwZhang/GaussianCube-Objaverse",
        "revision": "main",
        "model_path": "v1.1/objaverse_ckpt.pt",
        "mean_path": "v1.1/mean.pt",
        "std_path": "v1.1/std.pt",
        "bound": 0.5
    },
    "omniobject3d": {
        "repo_id": "BwZhang/GaussianCube-OmniObject3D-v1.0",
        "revision": "main",
        "model_path": "OmniObject3D_ckpt.pt",
        "mean_path": "mean.pt",
        "std_path": "std.pt",
        "bound": 1.0
    },
    "shapenet_car": {
        "repo_id": "BwZhang/GaussianCube-ShapeNetCar-v1.0",
        "revision": "main",
        "model_path": "shapenet_car_ckpt.pt", 
        "mean_path": "mean.pt",
        "std_path": "std.pt",
        "bound": 0.45
    },
    "shapenet_chair": {
        "repo_id": "BwZhang/GaussianCube-ShapeNetChair-v1.0",
        "revision": "main",
        "model_path": "shapenet_chair_ckpt.pt",
        "mean_path": "mean.pt",
        "std_path": "std.pt",
        "bound": 0.35
    }
}

def download_model_files(model_name):
    """Download model files from Hugging Face Hub."""
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REPOS.keys())}")
    
    model_info = MODEL_REPOS[model_name]
    downloaded_files = {}
    
    try:
        # Download model checkpoint
        downloaded_files["ckpt"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["model_path"],
            revision=model_info["revision"]
        )
        
        # Download mean file
        downloaded_files["mean"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["mean_path"],
            revision=model_info["revision"]
        )
        
        # Download std file
        downloaded_files["std"] = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["std_path"],
            revision=model_info["revision"]
        )
        
        downloaded_files["bound"] = model_info["bound"]
        
    except Exception as e:
        print(f"Error downloading files for {model_name}: {e}")
        raise
    
    return downloaded_files


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

    print(f"Downloading {args.model_name} files from Hugging Face...")
    downloaded_files = download_model_files(args.model_name)

    ckpt = downloaded_files["ckpt"]
    mean_file = downloaded_files["mean"]
    std_file = downloaded_files["std"]
    bound = downloaded_files["bound"]
    class_cond = args.model_name == "omniobject3d"
    text_cond = args.model_name == "objaverse_v1.1"
    cond_gen = text_cond or class_cond

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())
    seed_everything(args.seed + dist.get_rank())

    model_and_diffusion_config['model']['precision'] = "32"
    model = UNetModel(**model_and_diffusion_config['model'])

    diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    print("Loaded ckpt: ", ckpt)

    logger.configure(args.exp_name)
    options = logger.args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    model.to(dist_util.dev())
    model.eval()
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    clip_text_encoder = FrozenCLIPEmbedder()
    clip_text_encoder = clip_text_encoder.eval().to(dist_util.dev())
    if args.text:
        text_features = clip_text_encoder.encode(args.text)

    val_data = load_data(
        batch_size=1,
        deterministic=True,
        class_cond=class_cond,
        text_cond=text_cond,
    )

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(diffusion.betas).to(dist_util.dev()))
    std_volume = torch.tensor(init_volume_grid(bound=bound, num_pts_each_axis=32)).to(torch.float32).to(dist_util.dev()).contiguous()
    bg_color = torch.tensor([1, 1, 1]).to(torch.float32).to(dist_util.dev())
    mean = torch.load(mean_file).to(torch.float32).to(dist_util.dev())
    std = torch.load(std_file).to(torch.float32).to(dist_util.dev())

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
        if class_cond:
            condition['class_labels'] = model_kwargs['class_labels'].to(dist_util.dev())
            unconditional_condition['class_labels'] = torch.zeros_like(model_kwargs['class_labels']).to(dist_util.dev()) + 216
        
        if text_cond:
            condition['cond_text'] = text_features
            unconditional_condition['cond_text'] = torch.zeros_like(text_features)

        model_fn = model_wrapper(
            model,
            noise_schedule,
            model_type=MODEL_TYPES[model_and_diffusion_config['diffusion']['predict_type']],
            model_kwargs={},
            guidance_type='uncond' if not cond_gen else 'classifier-free',
            guidance_scale=args.guidance_scale,
            condition=None if not cond_gen else condition,
            unconditional_condition=None if not cond_gen else unconditional_condition,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')

        with torch.no_grad():
            noise = torch.randn(sample_shape, device=dist_util.dev()) * args.temperature

            samples = dpm_solver.sample(
                x=noise,
                steps=args.rescale_timesteps,
                t_start=1.0,
                t_end=1/1000,
                order=3 if not cond_gen else 2,
                skip_type='time_uniform',
                method='adaptive' if text_cond else 'multistep',
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
        
            samples = samples.permute(0, 2, 3, 4, 1).cpu()
            torch.save(samples[0], os.path.join(logger.get_dir(), f"rank_{dist.get_rank():02}_{img_id:04d}" + ".pt"))
            
            if args.render_video:
                s_path = os.path.join(logger.get_dir(), 'videos')
                os.makedirs(s_path,exist_ok=True)
                imageio.mimwrite(os.path.join(s_path, "rank_{:02}_render_{:06}.mp4".format(dist.get_rank(), img_id)), frames, fps=30)

        img_id += 1

 
def create_argparser():
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--model_name", type=str, required=True,
                       choices=list(MODEL_REPOS.keys()),
                       help="Name of the model to use")
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--seed", type=int, default=0)
    # Model config
    parser.add_argument("--config", type=str, default="configs/shapenet_uncond.yml")
    # Data args
    parser.add_argument("--active_sh_degree", type=int, default=0)
    # Inference args
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rescale_timesteps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--text", type=str, default="A donut with blue frosting and sprinkles.")
 
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
