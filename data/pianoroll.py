from .base import LatentDiscreteData
import torchvision
import torch
from pathlib import Path
from ldm.util import instantiate_from_config
import numpy as np

class Pianoroll(LatentDiscreteData):
    def __init__(self, root='data/pianoroll_dataset/pianoroll_dataset', train=True, num_samples=None):
        self.data = np.load(root+"/train.npy") if train else np.load(root+"/test.npy")
        self.data = torch.from_numpy(self.data).long()
        self.num_samples = num_samples

    def get_length(self):
        return 256
    
    def get_dim(self):
        return 129
    
    def __len__(self):
        return self.num_samples if self.num_samples is not None else len(self.data) 
    
    def __getitem__(self, i):
        return self.data[i]
