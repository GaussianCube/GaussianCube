import torch
import numpy as np
from model import gaussian_diffusion as gd
from model.respace import SpacedDiffusion, space_timesteps

from utils import dist_util


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_type="eps",
    predict_xstart=False, # Deprecated for compatibility
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    beta_start=0.0001,
    beta_end=0.02,
    min_snr=False,
    # offset_noise_level=0.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps, beta_start, beta_end)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    # if predict_xstart:
    #     model_mean_type = gd.ModelMeanType.START_X
    # else:
    #     model_mean_type = gd.ModelMeanType.EPSILON
    if predict_type == "eps":
        model_mean_type = gd.ModelMeanType.EPSILON
    elif predict_type == "xstart":
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_type == "v":
        model_mean_type = gd.ModelMeanType.V
    else:
        raise ValueError(f"Unknown predict_type for diffusion model: {predict_type}")
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        min_snr=min_snr,
    )


def predict_x0_from_q(diffusion, model, x0, t, clip_denoised=False, model_kwargs=None):
    t = torch.tensor([t, ]).long().to(x0.device).expand(x0.shape[0])
    xt = diffusion.q_sample(x0, t)
    pred = diffusion.p_mean_variance(
            model,
            xt,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
    return pred, xt


def init_volume_grid(bound=0.45, num_pts_each_axis=32):
    # Define the range and number of points  
    start = -bound
    stop = bound
    num_points = num_pts_each_axis  # Adjust the number of points to your preference  
    
    # Create a linear space for each axis  
    x = np.linspace(start, stop, num_points)  
    y = np.linspace(start, stop, num_points)  
    z = np.linspace(start, stop, num_points)  
    
    # Create a 3D grid of points using meshgrid  
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  
    
    # Stack the grid points in a single array of shape (N, 3)  
    xyz = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  
    
    return xyz


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_single_viewpoint_cam(cam_dict, idx):
    cam = {k: v[idx].to(dist_util.dev()).contiguous() for k, v in cam_dict.items()}
    return cam
