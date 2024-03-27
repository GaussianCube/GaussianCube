#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def parse_volume_data(volume, std_volume_xyz, active_sh_degree=0):
    """
    Parse volume data into a dictionary of tensors.
    volume of shape: (C, H, W, D)
    std_volume_xyz of shape: (H*W*D, 3)
    """
    sh_dim = 3 * ((active_sh_degree + 1) ** 2 - 1)
    C, H, W, D = volume.shape
    volume = (volume).permute(1, 2, 3, 0).reshape(-1, C)
    if torch.isnan(volume).any() or torch.isinf(volume).any():
        print("Data contains NaNs or Infs")
    xyz = (volume[:, :3] + std_volume_xyz)
    features_dc = volume[:, 3:6].reshape((xyz.shape[0], 3, 1)).transpose(1, 2)
    features_extra = volume[:, 6:6+sh_dim].reshape((xyz.shape[0], 3, (active_sh_degree + 1) ** 2 - 1)).transpose(1, 2)
    opacities = volume[:, 6+sh_dim:7+sh_dim].reshape((xyz.shape[0], 1)).clamp(0, 1)
    scales = volume[:, 7+sh_dim:10+sh_dim].reshape((xyz.shape[0], 3)).clamp(0, 1)
    rots = torch.nn.functional.normalize(volume[:, 10+sh_dim:].reshape((xyz.shape[0], 4)))
    if active_sh_degree > 0:
        shs = torch.cat([features_dc, features_extra], dim=1)
    else:
        shs = features_dc
    return {"xyz": xyz, "shs": shs, "opacities": opacities, "scales": scales, "rots": rots}


def render(viewpoint_camera, volume : torch.Tensor, std_volume : torch.Tensor, bg_color : torch.Tensor, active_sh_degree = 0, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    pc = parse_volume_data(volume, std_volume, active_sh_degree)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=active_sh_degree,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc["xyz"]
    means2D = screenspace_points
    opacity = pc["opacities"]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc["scales"]
    rotations = pc["rots"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc["shs"]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D.contiguous(),
        means2D = means2D.contiguous(),
        shs = shs.contiguous(),
        colors_precomp = colors_precomp,
        opacities = opacity.contiguous(),
        scales = scales.contiguous(),
        rotations = rotations.contiguous(),
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
