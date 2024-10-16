import os
import io
import socket

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm
from mpi4py import MPI
from PIL import Image
# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = torch.cuda.device_count()

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not torch.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    if not os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)
    port = comm.bcast(_find_free_port(), root=0)
    if not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dist.get_rank() % GPUS_PER_NODE}")
    return torch.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return torch.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


class DinoWrapper(nn.Module):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        if freeze:
            self._freeze()

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly sized
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False, do_resize=False).to(self.model.device)
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model(**inputs, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def _freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
            processor = ViTImageProcessor.from_pretrained(model_name)
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err


class ImageDataset(Dataset):
    def __init__(
        self,
        root_path,
        txt_file='',
        start_idx=0,
        end_idx=100,
        resolution=512,
        num_views=24,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    ):
        super().__init__()
        self.root_path = root_path
        self.txt_file = txt_file
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_views = num_views
        self.local_images = self.get_all_file()[shard:][::num_shards]
        self.resolution = resolution
        print("Total images: ", len(self.local_images))
    
    def get_all_file(self):
        with open(self.txt_file) as f:
            all_files = f.read().splitlines()[self.start_idx:self.end_idx]
        all_files = [os.path.join(obj, "imgs", f"timestep_{idx:02d}_view_00.png") for obj in all_files for idx in range(self.num_views)]
        todo_files = []
        for path in all_files:
            obj_name = path.split("/")[0]
            view_name = int(path.split("/")[2].replace(".png", "").split("_")[1])
            path_name = obj_name+"_timesteps_"+f"{view_name:02d}"
            if not os.path.exists(os.path.join(output_dir, path_name+".pt")):
                todo_files.append(path)

        return todo_files

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        full_path = os.path.join(self.root_path, path)
       
        try:
            pil_image2 = Image.open(full_path).resize((self.resolution, self.resolution))
            image = self.get_input_image_tensor(pil_image2)
        except:
            print("Error: ", full_path)
            return self.__getitem__(idx+1)

        obj_name = path.split("/")[0]
        view_name = int(path.split("/")[2].replace(".png", "").split("_")[1])
        data_dict = {"path": obj_name+"_timesteps_"+f"{view_name:02d}"}
        return image, data_dict
    
    def get_input_image_tensor(self, image):
        im_data = np.array(image.convert("RGBA")).astype(np.float32)
        bg = np.array([1,1,1]).astype(np.float32)
        norm_data = im_data / 255.0
        alpha = norm_data[:, :, 3:4]
        blurred_alpha = alpha
        arr = norm_data[:,:,:3] * blurred_alpha + bg * (1 - blurred_alpha)
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return image_tensor


setup_dist()

batch_size = 8
dataset = ImageDataset(root_path='./example_data/avatar/', txt_file='./example_data/avatar.txt', start_idx=0, end_idx=32)  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)  

model = DinoWrapper('facebook/dino-vitb16').to(dev())

# Forward the data in batches and save the encoded latent tensors  
output_dir = "./example_data/avatar/avatar_dino_feature"   
os.makedirs(output_dir, exist_ok=True)  
  
for i, (images, data_dicts) in enumerate(tqdm(dataloader)):  
    images = images.to(dev())  
  
    with torch.no_grad():  
        latent = model(images)
  
    for j, latent_sample in enumerate(latent):  
        output_path = os.path.join(output_dir, data_dicts["path"][j] + ".pt")  
        torch.save(latent_sample.cpu(), output_path)
