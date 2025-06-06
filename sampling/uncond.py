from .base import Algo
import torch
import tqdm
from .sampling_utils import get_pc_sampler

class Uncond(Algo):
    def __init__(self, net, forward_op, device='cuda', latent=False):
        super().__init__(net, forward_op)
        self.device = device
        self.latent = latent

    def inference(self, num_samples=1, observation=None, verbose=False):
        uncond_sampler = get_pc_sampler(self.net.graph, self.net.noise, (num_samples, self.net.length), 'analytic', 128, device=self.device)
        x = uncond_sampler(self.net)
        if self.latent:
            x = self.net.decode(x)
        return x
