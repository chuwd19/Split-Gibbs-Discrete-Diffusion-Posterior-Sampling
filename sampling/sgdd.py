from .base import Algo
import torch
import tqdm
import numpy as np
from .sampling_utils import get_pc_sampler
import os

class SGDD(Algo):
    """
        Implementation of split Gibbs sampling for discrete diffusion.
        https://arxiv.org/abs/2405.18782 (continuous version)
    """

    def __init__(self, net, forward_op, num_steps=200, ode_steps=128, eps=1e-5, mh_steps=1000, alpha=1, max_dist = 1, device='cuda', cf=False):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__(net=net, forward_op=forward_op)

        # self.model = self.net.model
        self.graph = self.net.graph
        self.noise = self.net.noise
        self.uncond_sampler = get_pc_sampler(self.graph, self.noise, (1,1024), 'analytic', ode_steps , device=device)
        self.device = device
        self.num_steps = num_steps
        self.sigma_fn = lambda t: t
        self.get_time_step_fn = lambda r: (1 + r * (eps  - 1))
        steps = torch.linspace(0, 1-eps, num_steps)
        # steps = torch.linspace(0.5, 0.7, num_steps)
        # p = 0.245 # sigma=1
        # p = 0.302 # sigma=0.5
        # p = 0.379   # sigma=0.2
        # p = 0.434   # sigma=0.1
        # p = 0.49  # sigma=0.05
        # p = 0.61  # sigma=0.01
        # p = 0.68  # sigma=0.005
        # p = 0.8  # sigma=0.001
        # steps = torch.ones(num_steps) * p # for ablation study!
        # print(self.noise(1-p)[0])
        
        self.time_steps = self.get_time_step_fn(steps)
        self.ode_steps = ode_steps
        self.mh_steps = mh_steps
        self.alpha = alpha
        self.max_dist = max_dist
        self.cf = cf

    def log_ratio(self, sigma, hm_dist):
        
        alpha = (1 - np.exp(-sigma)) * (1 - 1/self.graph.dim)
        log_alpha = np.log(alpha+1e-5)
        log_1alpha = np.log(1 - alpha)
        log_ratio = hm_dist * log_alpha + (self.net.length - hm_dist) * log_1alpha
        return log_ratio
    
    def metropolis_hasting(self, x0hat, op, y, sigma, steps):
        x = x0hat.clone()
        dim = self.graph._dim
        N, L = x0hat.shape[0], x0hat.shape[1]
        current_log_likelihood = op.log_likelihood(x, y)
        current_hm_dist = (x != x0hat).sum(dim=-1)
        for _ in range(steps):

            # Get proposal
            
            for _ in range(self.max_dist):
                proposal = x.clone() # proposal, shape = [N, L]
                # for _ in range(self.max_dist):
                idx = torch.randint(L, (N,), device=x.device)
                v = torch.randint(dim, (N,), device=x.device)
                proposal.scatter_(1, idx[:, None], v.unsqueeze(1))

            # Compute log prob difference
            log_likelihood = op.log_likelihood(proposal,y)
            hm_dist = (proposal != x0hat).sum(dim=-1)
            log_ratio = log_likelihood - current_log_likelihood
            log_ratio += self.log_ratio(sigma, hm_dist) - self.log_ratio(sigma, current_hm_dist)

            # Metropolis-Hasting step
            rho = torch.clip(torch.exp(log_ratio), max=1.0)
            seed = torch.rand_like(rho)
            x = x * (seed > rho).unsqueeze(-1) + proposal * (seed < rho).unsqueeze(-1)
            current_log_likelihood = log_likelihood * (seed < rho)+ current_log_likelihood * (seed > rho)
            current_hm_dist = hm_dist * (seed < rho) + current_hm_dist * (seed > rho)
            
        return x

    @torch.no_grad()
    def inference(self, observation=None, num_samples=1, verbose=True):

        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        x_start = self.graph.sample_limit(num_samples, self.net.length).to(self.device)
        
        xt = x_start.to(self.device)
        
        # x0hats, xts = [], []
        
        for i in pbar:

            # 1. reverse diffusion
            x0hat = self.uncond_sampler(self.net, xt, t_start=self.time_steps[i])
            
            # 2. Metropolis-Hasting
            sigma, _ = self.noise(self.time_steps[i])
            if self.cf:
                x0y = self.forward_op.cf_sample(x0hat, observation, sigma*self.alpha)
            else:
                x0y = self.metropolis_hasting(x0hat, self.forward_op, observation, sigma*self.alpha, steps=self.mh_steps)
            xt = x0y

            # x0hats.append(x0hat)
            # xts.append(xt)
        # path = "analysis/ablation/sigma_0.001"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        
        # torch.save(torch.cat(x0hats, dim=0), os.path.join(path,'x0hats.pt'))
        # torch.save(torch.cat(xts, dim=0), os.path.join(path,'xts.pt'))
        return xt
    
    
class SGDD_latent(SGDD):
    def metropolis_hasting(self, x0hat, op, y, sigma, steps):
        x = x0hat.clone()
        dim = self.graph._dim
        N, L = x0hat.shape[0], x0hat.shape[1]
        ## decode x:
        x_decoded = self.net.decode(x)
        current_log_likelihood = op.log_likelihood(x_decoded, y)
        current_hm_dist = (x != x0hat).sum(dim=-1)
        for _ in range(steps):
            for _ in range(self.max_dist):
                proposal = x.clone() # proposal, shape = [N, L]
                # for _ in range(self.max_dist):
                idx = torch.randint(L, (N,), device=x.device)
                v = torch.randint(dim, (N,), device=x.device)
                proposal.scatter_(1, idx[:, None], v.unsqueeze(1))
            proposal_decoded = self.net.decode(proposal)
            log_likelihood = op.log_likelihood(proposal_decoded,y)
            hm_dist = (proposal != x0hat).sum(dim=-1)
            log_ratio = log_likelihood - current_log_likelihood
            log_ratio += self.log_ratio(sigma, hm_dist) - self.log_ratio(sigma, current_hm_dist)
            rho = torch.clip(torch.exp(log_ratio), max=1.0)
            seed = torch.rand_like(rho)
            x = x * (seed > rho).unsqueeze(-1) + proposal * (seed < rho).unsqueeze(-1)
            current_log_likelihood = log_likelihood * (seed < rho)+ current_log_likelihood * (seed > rho)
            current_hm_dist = hm_dist * (seed < rho) + current_hm_dist * (seed > rho)
            
        return x
    
    def inference(self, observation=None, num_samples=1, verbose=True):
        z = super().inference(observation, num_samples, verbose)
        return self.net.decode(z)