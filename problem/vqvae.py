from .base import BaseOperator
import torch
import numpy as np
from utils.resizer import Resizer

class SuperResolution(BaseOperator):
    def __init__(self,resolution=256, factor=4, sigma_noise=0.01, device='cuda'):
        super().__init__(sigma_noise)
        self.resolution = resolution
        self.kernel = Resizer([1,3,resolution,resolution], 1/factor).to(device)

    def __call__(self, inputs):
        return self.kernel(inputs)

    def loss(self, inputs, y):
        return ((self(inputs) - y) ** 2).flatten(1).sum(-1)

    def reward(self, inputs, y):
        return -self.loss(inputs, y)

    def log_likelihood(self, inputs, y):
        return -self.loss(inputs, y)/self.sigma_noise**2
    
class InpaintRand(BaseOperator):
    def __init__(self, resolution=256, ratio=0.7, sigma_noise=0.01, device='cuda'):
        super().__init__(sigma_noise)
        self.resolution = resolution
        self.ratio = ratio
        self.mask = torch.ones((1, resolution*resolution), device=device)
        samples = np.random.choice(self.resolution**2, int(self.resolution**2 *ratio), replace=False)
        self.mask[:, samples] = 0
        self.mask = self.mask.view(1, self.resolution, self.resolution).repeat(1, 3, 1, 1)

    def __call__(self, inputs):
        return inputs * self.mask

    def loss(self, inputs, y):
        return ((self(inputs) - y) ** 2).flatten(1).sum(-1)

    def reward(self, inputs, y):
        return -self.loss(inputs, y)

    def log_likelihood(self, inputs, y):
        return -self.loss(inputs, y)/self.sigma_noise**2