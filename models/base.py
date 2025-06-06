from abc import ABC, abstractmethod
import torch

class DiscreteDiffusion(ABC):
    def __init__(self):
        pass

    def score(self, x, sigma):
        pass

    def get_start(self, batch_size):
        pass
    
    @property
    def dim(self):
        pass
    
    @property
    def length(self):
        pass
    
    def pred_mean(self, x, t):
        pass



    def q_sample(self, x, t):
        pass