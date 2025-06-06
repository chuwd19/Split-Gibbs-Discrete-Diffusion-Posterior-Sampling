from .base import BaseOperator
import torch

class XOR(BaseOperator):
    def __init__(self, ratio, length, **kwargs):
        super().__init__()
        self.length = length
        self.ratio = ratio
        torch.manual_seed(0)
        self.pairs = torch.randint(0, length, (2,int(ratio*length)))

    def __call__(self, inputs):
        return inputs[:, self.pairs[0]] ^ inputs[:, self.pairs[1]]
    
    def loss(self, inputs, y):
        # hamming distance
        return ((self(inputs) != y)).flatten(1).sum(-1)
    
class AND(BaseOperator):
    def __init__(self, ratio, length, **kwargs):
        super().__init__()
        self.length = length
        self.ratio = ratio
        torch.manual_seed(0)
        self.pairs = torch.randint(0, length, (2,int(ratio*length)))

    def __call__(self, inputs):
        return inputs[:, self.pairs[0]] & inputs[:, self.pairs[1]]
    
    def loss(self, inputs, y):
        # hamming distance
        return ((self(inputs) != y)).flatten(1).sum(-1)
    
    
class Inpaint(BaseOperator):
    def __init__(self, ratio, length, **kwargs):
        super().__init__()
        self.length = length
        self.ratio = ratio
        torch.manual_seed(0)
        self.mask = (torch.rand(length) < ratio).long()
        
    def __call__(self, inputs):
        return inputs * self.mask.to(inputs.device)
    
    def loss(self, inputs, y):
        # hamming distance
        return ((self(inputs) != y)).flatten(1).sum(-1)
    
    def cf_sample(self, inputs, y, sigma):
        prob = torch.exp(-sigma)
        mask = (torch.rand(inputs.shape) < prob).long()
        
        measured =  inputs * mask.to(inputs.device) + y * (1 - mask.to(inputs.device))
        return inputs * (1 - self.mask.to(inputs.device)) + measured * self.mask.to(inputs.device) 

class InpaintBox(BaseOperator):
    def __init__(self, length, **kwargs):
        super().__init__()
        self.length = length
        self.mask = torch.ones(32,32)
        self.mask[5:20, 5:25] = 0
        self.mask = self.mask.flatten()
        
    def __call__(self, inputs):
        return inputs * self.mask.to(inputs.device)
    
    def loss(self, inputs, y):
        # hamming distance
        return ((self(inputs) != y)).flatten(1).sum(-1)
    
    def cf_sample(self, inputs, y, sigma):
        prob = torch.exp(-sigma)
        mask = (torch.rand(inputs.shape) < prob).long()
        
        measured =  inputs * mask.to(inputs.device) + y * (1 - mask.to(inputs.device))
        return inputs * (1 - self.mask.to(inputs.device)) + measured * self.mask.to(inputs.device)
    
class SingleModal(BaseOperator):
    def __init__(self, length, grid_solution=10, device='cuda', sigma_noise=3):
        super().__init__(sigma_noise=sigma_noise, device=device)
        self.length = length
        self.dim = grid_solution
        self.center = torch.ones(length, device=device) * (self.dim-1)/2
    def __call__(self, inputs):
        return (inputs - self.center).abs()#.sum(-1)
    
    def loss(self, inputs, y):
        return (((self(inputs) - y)) ** 2).mean(-1)
