import copy
import os
import time
import glob
import imageio
import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from utils import dist_util, logger
from utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)

from utils.script_util import init_volume_grid
from model.nn import update_ema
from model.resample import UniformSampler
from model.gaussian_diffusion import ModelMeanType
from gaussian_renderer import render
from utils.lpips.lpips import LPIPS
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
 

class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        uncond_p=1.0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        active_sh_degree=0,
        white_background=True,
        use_vgg=False,
        use_tensorboard=True,
        render_lpips_weight=1.0,
        render_l1_weight=1.0,
        mean_file=None,
        std_file=None,
        bound=0.45,
        has_pretrain_weight=False,
        diffusion_loss_weight=1.0,
        num_pts_each_axis=32,
        dataset_type='shapenet',
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = find_resume_checkpoint(resume_checkpoint)
        self.uncond_p = uncond_p
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.active_sh_degree = active_sh_degree
        self.bg_color = th.tensor([1,1,1]).to(th.float32).to(dist_util.dev()) if white_background else th.tensor([0,0,0]).to(th.float32).to(dist_util.dev())
        self.std_volume = th.tensor(init_volume_grid(bound=bound, num_pts_each_axis=num_pts_each_axis)).to(th.float32).to(dist_util.dev()).contiguous()
        self.has_pretrain_weight = has_pretrain_weight
        self.diffusion_loss_weight = diffusion_loss_weight
        self.dataset_type = dataset_type

        if use_vgg:
            self.vgg = LPIPS(net_type='vgg').to(dist_util.dev()).eval()
            self.L1loss = th.nn.L1Loss()
            self.render_lpips_weight = render_lpips_weight
            self.render_l1_weight = render_l1_weight
            print('Using VGG loss')
        else:
            self.L1loss = None
            self.vgg = None
            self.render_lpips_weight = 0.
            self.render_l1_weight = 0.
        
        if mean_file is not None and std_file is not None:
            self.mean = th.load(mean_file).to(th.float32).to(dist_util.dev())
            self.std = th.load(std_file).to(th.float32).to(dist_util.dev())
            if len(self.mean.shape) == 1:
                # Mean and scale of shape (C,) for conditional generation
                print("Using mean shape: ", self.mean.shape)
                self.mean = self.mean.reshape(1, -1, 1, 1, 1).requires_grad_(False).contiguous()
                self.std = self.std.reshape(1, -1, 1, 1, 1).requires_grad_(False).contiguous()
            else:
                # Mean and std of shape (H, W, D, C) for unconditional generation
                print("Using mean shape: ", self.mean.shape)
                self.mean = self.mean.permute(3, 0, 1, 2).requires_grad_(False).unsqueeze(0).contiguous()
                self.std = self.std.permute(3, 0, 1, 2).requires_grad_(False).unsqueeze(0).contiguous()
        else:
            self.mean = th.tensor([0]).to(th.float32).to(dist_util.dev())
            self.std = th.tensor([1]).to(th.float32).to(dist_util.dev())
        
        self.optimize_model = self.model
        self.model_params = list(self.optimize_model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        num_warmup_steps = 1000
        def warmup_lr_schedule(steps):  
            if steps < num_warmup_steps:  
                return float(steps) / float(max(1, num_warmup_steps))  
            return 1.0  
        
        # Apply this schedule to the optimizer using LambdaLR  
        self.warmup_scheduler = th.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=warmup_lr_schedule)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and dist.get_rank() == 0:
            self.writer = logger.Visualizer(os.path.join(logger.get_dir(), 'tf_events'))

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            print("resume checkpoint: ", resume_checkpoint)
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(th.load(resume_checkpoint, map_location="cpu"),strict=False)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location="cpu")
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step <= self.lr_anneal_steps
        ):
            batch, model_kwargs = next(self.data)
            self.run_step(batch, model_kwargs)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
         
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, model_kwargs):
        start_time = time.time()
        self.forward_backward(batch, model_kwargs)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        step_time = time.time() - start_time
        logger.logkv_mean("step_time", step_time)
        self.log_step()
    
    def get_pred_x0(self, output, t):
        if self.diffusion.model_mean_type == ModelMeanType.START_X:
            pred_x0 = output['model_output'] 
        elif self.diffusion.model_mean_type == ModelMeanType.V:
            pred_x0 = self.diffusion._predict_start_from_z_and_v(x_t=output['x_t'], t=t, v=output['model_output'])
        else:
            pred_x0 = self.diffusion._predict_xstart_from_eps(output['x_t'], t, output['model_output'])
        return pred_x0

    def forward_backward(self, batch, model_kwargs):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {n: model_kwargs[n][i:i+self.microbatch].to(dist_util.dev()) for n in model_kwargs if n in ['cond_text', 'class_labels']}
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, _ = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            if last_batch or not self.use_ddp:
                losses, output = self.diffusion.training_losses(self.ddp_model, micro, t, micro_cond)
            else:
                with self.ddp_model.no_sync():
                    losses, output = self.diffusion.training_losses(self.ddp_model, micro, t, micro_cond)
            loss = losses["loss"].mean() * self.diffusion_loss_weight

            self.has_pretrain_weight = self.has_pretrain_weight or self.step >= 50_000 or self.resume_step > 0
            pred_x0_denorm = None
            if self.vgg and "cams" in model_kwargs and self.has_pretrain_weight:
                pred_x0 = self.get_pred_x0(output, t)
                pred_x0_denorm = pred_x0 * self.std + self.mean
                pred_imgs, gt_imgs = [], []
                for cam_idx, cam_info in enumerate(model_kwargs["cams"]):
                    for b in range(micro.shape[0]):
                        cam = build_single_viewpoint_cam(cam_info, b)
                        res = render(cam, pred_x0_denorm[b], self.std_volume, self.bg_color, self.active_sh_degree)
                        pred_imgs.append(res["render"])
                        gt_imgs.append(cam["image"])
                pred_img = th.stack(pred_imgs, dim=0)
                gt_img = th.stack(gt_imgs, dim=0)
                pixel_l1_loss = self.L1loss(pred_img, gt_img) * self.render_l1_weight
                vgg_loss = self.vgg(pred_img*2 - 1., gt_img*2 - 1.) * self.render_lpips_weight
                losses["pixel_l1_loss"] = th.tensor([pixel_l1_loss])
                losses["vgg_loss"] = th.tensor([vgg_loss])
                loss = loss + pixel_l1_loss + vgg_loss
                
                if self.step % 100 == 0 and dist.get_rank() == 0:
                    s_path = os.path.join(logger.get_dir(), 'train_images')
                    os.makedirs(s_path,exist_ok=True)
                    output_image = pred_img[0].clamp(0.0, 1.0)
                    rgb_map = output_image.squeeze().permute(1, 2, 0).cpu() 
                    rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                    imageio.imwrite(os.path.join(s_path, "render_iter_{:08}_t_{:02}.png".format(self.step, int(t[-1]))), rgb_map)

                    rgb_map = gt_img[0].squeeze().permute(1, 2, 0).cpu() 
                    rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                    imageio.imwrite(os.path.join(s_path, "gt_iter_{:08}_t_{:02}.png".format(self.step, int(t[-1]))), rgb_map)
            
            log_loss_dict(
                self.diffusion, t, {k: v for k, v in losses.items()}
            )
            if self.use_tensorboard and self.step % self.log_interval == 0 and dist.get_rank() == 0:
                if self.vgg:
                    self.writer.write_dict({"loss": loss.item(), "pixel_l1_loss": losses["pixel_l1_loss"].item() if "pixel_l1_loss" in losses else 0., "vgg_loss": losses["vgg_loss"].item() if "vgg_loss" in losses else 0.}, self.step)
                else:
                    self.writer.write_dict({"loss": loss.item()}, self.step)
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        self.warmup_scheduler.step()
        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        nn.utils.clip_grad_norm_(self.model_params, 1.0)
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        self.warmup_scheduler.step()
        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        for name, param in self.optimize_model.named_parameters():
            if param.grad==None:
                print(name, )

        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        return

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with open(os.path.join(get_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with open(
                os.path.join(get_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.optimize_model.parameters()), master_params
            )
        state_dict = self.optimize_model.state_dict()
        for i, (name, _value) in enumerate(self.optimize_model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.optimize_model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    filename=filename.split('/')[-1]
    assert(filename.endswith(".pt"))
    filename=filename[:-3]
    if filename.startswith("model"):
        split = filename[5:]
    elif filename.startswith("ema"):
        split = filename.split("_")[-1]
    else:
        return 0
    try:
        return int(split)
    except ValueError:
        return 0


def get_logdir():
    p = os.path.join(logger.get_dir(),"checkpoints")
    os.makedirs(p,exist_ok=True)
    return p


def find_resume_checkpoint(resume_checkpoint):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if not resume_checkpoint:
        return None
    if "ROOT" in resume_checkpoint:
        maybe_root=os.environ.get("AMLT_MAP_INPUT_DIR")
        maybe_root="OUTPUT/log" if not maybe_root else maybe_root
        root=os.path.join(maybe_root,"checkpoints")
        resume_checkpoint=resume_checkpoint.replace("ROOT",root)
    if "LATEST" in resume_checkpoint:
        files=glob.glob(resume_checkpoint.replace("LATEST","*.pt"))
        if not files:
            return None
        return max(files,key=parse_resume_step_from_filename)
    return resume_checkpoint


def build_single_viewpoint_cam(cam_dict, idx):
    cam = {k: v[idx].to(dist_util.dev()).contiguous() for k, v in cam_dict.items()}
    return cam

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

