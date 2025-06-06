from .base import DiscreteData
import torchvision
import torch

class MNIST(DiscreteData):
    def __init__(self, train=True, discrete=True, num_samples=None):
        dataset = torchvision.datasets.MNIST(root='data', train=train, download=True, transform=None)
        padding = torchvision.transforms.Pad(2)
        self.data = dataset.data
        self.data = padding(self.data)
        self.label = dataset.targets
        self.discrete = discrete
        self.num_samples = num_samples

    def compress(self, img):
        scale = 256 / self.get_dim()
        return (img / scale).int().flatten(-2)
    
    def decompress(self, img):
        scale = 256 / self.get_dim()
        return (img * scale).float().reshape(-1, 1, 32, 32)
    
    def normalize(self, img):
        return img.float() / 256.0 
     
    def get_length(self):
        return 1*32*32

    def get_dim(self):
        return 2

    def __len__(self):
        return self.num_samples if self.num_samples is not None else len(self.data)
    
    def __getitem__(self, i):
        if self.discrete:
            return self.compress(self.data[i]).to(torch.int64)
        return self.decompress(self.compress(self.data[i]).to(torch.int64))/128, self.label[i]
