from .base import LatentDiscreteData
import torchvision
import torch
from pathlib import Path
from ldm.util import instantiate_from_config
from PIL import Image
from omegaconf import OmegaConf

class VQFFHQ(LatentDiscreteData):
    def __init__(self, root='dataset/demo', encoder_config='configs/vqvae/ffhq_f8_n256.yaml', first_stage_model="checkpoints/vq-f8-n256.ckpt", 
                 resolution=256, device='cuda', f = 8, start_id=None, end_id=None):
        # Define the file extensions to search for
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.data = [file for ext in extensions for file in Path(root).rglob(ext)]
        self.data = sorted(self.data)
        
        # Subset the dataset
        self.data = self.data[start_id: end_id]
        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(resolution),
            torchvision.transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device
        cfg = OmegaConf.load(encoder_config)
        self.first_stage_model_path = first_stage_model
        self.instantiate_first_stage(cfg)
        self.f = f
        
    def get_length(self):
        return int(self.res**2 / self.f**2)
    
    def get_dim(self):
        return self.first_stage_model.quantize.n_embed
    
    def instantiate_first_stage(self, config):
        # From "https://github.com/CompVis/latent-diffusion"
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval().to(self.device)
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        model.load_state_dict(torch.load(self.first_stage_model_path, map_location=self.device)['state_dict'])

    def encode(self, img):
        # Encode as indices
        return self.first_stage_model.encode(img)[2][2].reshape(-1, self.get_length())
    
    def decode(self, z):
        # Decode from indices
        z = z.reshape(-1, self.res//self.f, self.res//self.f)
        z = self.first_stage_model.quantize.embedding(z).permute(0,3,1,2)
        return self.first_stage_model.decode(z)

    def __getitem__(self, i):
        img = (self.trans(Image.open(self.data[i])) * 2 - 1)
        if img.shape[0] == 1:
            img = torch.cat([img] * 3, dim=0)
        return img

    def __len__(self):
        return len(self.data)
