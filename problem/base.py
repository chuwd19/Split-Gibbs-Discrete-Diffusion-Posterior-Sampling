from abc import ABC, abstractmethod
from torch.autograd import grad

import torch


class BaseOperator(ABC):
    '''
    Base class for forward operators.
    Settings:
        1. Reward function: R(x) = - loss(x)
        2. Inverse problem: R(x) = - loss(x,y) = - log p(y|x) * sigma_noise^2
    '''
    def __init__(self, sigma_noise=0.01, device='cuda'):
        self.sigma_noise = sigma_noise
        self.device = device

    @abstractmethod
    def __call__(self, inputs, **kwargs):
        '''
        inputs : torch.tensor with shape (batch_size, ...). 
        1. Reward function: return loss(x)
        2. Inverse problem: return y = A(x)
        '''
        pass
    
    @abstractmethod
    def loss(self, inputs, y=None, **kwargs):
        '''
        inputs : torch.tensor with shape (batch_size, ...). 
                 Note that inputs have been normalized to the input range of pre-trained diffusion models.
        y : torch.tensor with shape (batch_size, ...).
            Note that y have been normalized to the input range of pre-trained diffusion models.
        '''
        pass

    def reward(self, inputs, y=None, **kwargs):
        return -self.loss(inputs, y, **kwargs)
    
    def log_likelihood(self, inputs, y=None, **kwargs):
        '''
        x : torch.tensor with shape (batch_size, ...). 
            Note that x have been normalized to the input range of pre-trained diffusion models.
        y : torch.tensor with shape (batch_size, ...).
            Note that y have been normalized to the input range of pre-trained diffusion models.
        '''
        return -self.loss(inputs, y, **kwargs)/self.sigma_noise**2