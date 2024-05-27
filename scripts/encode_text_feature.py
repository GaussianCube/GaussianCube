import os
import io
import json
import socket

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from mpi4py import MPI
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


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class ImageDataset(Dataset):
    def __init__(
        self,
        txt_file='',
        start_idx=0,
        end_idx=100,
        resolution=512,
        num_views=10,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    ):
        super().__init__()
        self.txt_file = txt_file
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_views = num_views
        self.local_images = self.get_all_file()[shard:][::num_shards]
        self.resolution = resolution
        with open(self.txt_file) as f:
            self.json_file = json.load(f)
        print("Total files: ", len(self.local_images))

    def get_all_file(self):
        with open(self.txt_file) as f:
            all_files = sorted(json.load(f).keys())[self.start_idx:self.end_idx]
        return all_files

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        text = self.json_file[path]
        data_dict = {"text": text, "path": path}
        return data_dict

setup_dist()

batch_size = 32
dataset = ImageDataset(txt_file='./example_data/objaverse_captions.json', start_idx=0, end_idx=32)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Forward the data in batches and save the encoded latent tensors
output_dir = "./example_data/objaverse/objaverse_text_feature"
os.makedirs(output_dir, exist_ok=True)

model = FrozenCLIPEmbedder()
model = model.eval().to(dev())
for i, (data_dicts) in enumerate(tqdm(dataloader)):

    with torch.no_grad():
        latent = model.encode(data_dicts["text"])

    for j, latent_sample in enumerate(latent):
        output_path = os.path.join(output_dir, data_dicts["path"][j] + ".pt")
        torch.save(latent_sample.cpu(), output_path)
