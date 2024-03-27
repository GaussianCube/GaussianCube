import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from utils.script_util import init_volume_grid, build_rotation
from kiui.cam import orbit_camera


def load_data(
    *,
    batch_size,
    deterministic=False,
    class_cond=False,
):
    dataset = InferenceDataset(
        class_cond=class_cond,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True
    )
    while True:
        yield from loader


class InferenceDataset(Dataset):
    def __init__(
        self,
        class_cond=False,
    ):
        super().__init__()
        self.class_cond = class_cond
        self.num_classes = 216

    def __len__(self):
        return  10000000000

    def __getitem__(self, idx):
        data_dict = {}

        if self.class_cond:
            classes = list(range(0, self.num_classes))
            random.shuffle(classes)
            class_label = torch.tensor(classes[0], dtype=torch.long)
            data_dict["class_labels"] = class_label

        azimuth = np.arange(0, 360, 6, dtype=np.int32)
        elevation = -30
        cam_radius = 1.2 if not self.class_cond else np.sqrt(16.25)
        cams = []
        convert_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(np.float32)
        for azi in azimuth:
            cam_poses = orbit_camera(elevation, azi, radius=cam_radius, opengl=True)
            cam_poses = convert_mat @ cam_poses
            cams.append(load_cam(c2w=cam_poses, class_cond=self.class_cond))
        data_dict["cams"] = cams
        return data_dict
        

def load_cam(c2w, class_cond=False):
    fovx = 0.8575560450553894 if not class_cond else 0.6911112070083618
    orig_image_size = 512
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    fovy = focal2fov(fov2focal(fovx, orig_image_size), orig_image_size)
    
    R = R
    T = T
    FoVx = fovx
    FoVy = fovy

    image_width = orig_image_size
    image_height = orig_image_size

    zfar = 100.0
    znear = 0.01

    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0

    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return {"FoVx": fovx, "FoVy": fovy, "image_width": orig_image_size, "image_height": orig_image_size, "world_view_transform": world_view_transform, "projection_matrix": projection_matrix, "full_proj_transform": full_proj_transform, "camera_center": camera_center, "c2w": c2w}


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
