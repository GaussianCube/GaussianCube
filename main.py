import argparse
import torch
import torch.distributed as dist
import torch.utils.cpp_extension
from omegaconf import OmegaConf

from model.unet import UNetModel
from model.resample import UniformSampler
from utils import dist_util, logger
from utils.script_util import create_gaussian_diffusion
from train import TrainLoop

def main():
    args = create_argparser().parse_args()

    model_and_diffusion_config = OmegaConf.load(args.config)
    print("Model and Diffusion config: ", model_and_diffusion_config)

    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())

    model = UNetModel(**model_and_diffusion_config['model'])
    diffusion = create_gaussian_diffusion(**model_and_diffusion_config['diffusion'])
    has_pretrain_weight = False
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        has_pretrain_weight = True

    logger.configure(args.exp_name)
    options = logger.args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    model.to(dist_util.dev())
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    schedule_sampler = UniformSampler(model_and_diffusion_config['diffusion']['steps'])

    logger.log("creating data loader...")

    dataset_type = 'omni' if args.omni else 'objaverse' if args.objaverse else 'shapenet'
    if args.omni:
        from dataset.dataset_omni import load_data
    elif args.objaverse:
        from dataset.dataset_objaverse import load_data
    else:
        from dataset.dataset import load_data

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=model_and_diffusion_config['model']['image_size'],
        train=True,
        uncond_p=args.uncond_p,
        mean_file=args.mean_file,
        std_file=args.std_file,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        txt_file=args.txt_file,
        load_camera=args.load_camera,
        cam_root_path=args.cam_root_path,
        clip_input=args.clip_input,
        bound=args.bound,
        text_feature_root=args.text_feature_root,
    )

    logger.log("training...")
    TrainLoop(
        model,
        diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        uncond_p=args.uncond_p,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        schedule_sampler=schedule_sampler,
        use_vgg=args.use_vgg,
        use_tensorboard=args.use_tensorboard,
        render_l1_weight=args.render_l1_weight,
        render_lpips_weight=args.render_lpips_weight,
        mean_file=args.mean_file,
        std_file=args.std_file,
        bound=args.bound,
        has_pretrain_weight=has_pretrain_weight,
        diffusion_loss_weight=args.diffusion_loss_weight,
        num_pts_each_axis=args.num_pts_each_axis,
        dataset_type=dataset_type,
    ).run_loop()

 
def create_argparser():
    def none_or_str(value):  
        if value.lower() == 'none':  
            return None  
        return value
    parser = argparse.ArgumentParser()
    # Experiment args
    parser.add_argument("--exp_name", type=str, default="/tmp/output/")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    # Model config
    parser.add_argument("--config", type=str, default="configs/shapenet_uncond.yml")
    # Train args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--use_vgg", action="store_true")
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--uncond_p", type=float, default=1.0)
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0)
    parser.add_argument("--render_l1_weight", type=float, default=1.0)
    parser.add_argument("--render_lpips_weight", type=float, default=1.0)
    # Data args
    parser.add_argument("--data_dir", type=str, default="./example_data/shapenet/volume_act/")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument("--txt_file", type=str, default="./example_data/shapenet/shapenet_train.txt")
    parser.add_argument("--mean_file", type=none_or_str, default="./example_data/shapenet/mean_volume_act.pt")
    parser.add_argument("--std_file", type=none_or_str, default="./example_data/shapenet/std_volume_act.pt")
    parser.add_argument("--load_camera", type=int, default=0)
    parser.add_argument("--cam_root_path", type=str, default="./example_data/shapenet/shapenet_rendering_512/")
    parser.add_argument("--bound", type=float, default=0.45)
    parser.add_argument("--omni", action="store_true")
    parser.add_argument("--objaverse", action="store_true")
    parser.add_argument("--clip_input", action="store_true")
    parser.add_argument("--text_feature_root", type=str, default="./example_data/objaverse/objaverse_text_feature/")
    parser.add_argument("--num_pts_each_axis", type=int, default=32)
 
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
