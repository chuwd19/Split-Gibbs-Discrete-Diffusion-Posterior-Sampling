from .base import DiscreteDiffusion
from .SEDD.load_model import load_model
from ldm.util import instantiate_from_config

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    
class SEDD(DiscreteDiffusion):
    def __init__(self, model_path, device='cuda'):
        model, graph, noise = load_model(model_path, device)
        self.model = model
        self.graph = graph
        self.noise = noise
        self.device = device

    def score(self, x, sigma):
        # sigma, dsigma = self.noise(t)
        return self.model(x, sigma.reshape(-1))
    
    def pred_mean(self, x, t):
        # This could be bad since score is modeled independently for each dimension
        # according to the paper (Theorem 4.1).
        sigma, dsigma = self.noise(t)
        score = self.model(x, sigma)

        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        return sample_categorical(probs)

    def get_start(self, batch_size):
        return self.graph.sample_limit((batch_size,)).to(self.device)
    
    @property
    def dim(self):
        return self.graph.dim
    
    @property
    def length(self):
        return self.model.length
    
def load_first_stage_model(config,device='cuda'):
    if config.name == 'vqvae':
        cfg = OmegaConf.load(config.config)
        first_stage_model = instantiate_from_config(cfg).eval()
        for param in first_stage_model.parameters():
            param.requires_grad = False
        first_stage_model.load_state_dict(torch.load(config.path)['state_dict'])
        first_stage_model = first_stage_model.to(device)
        size = config.res // config.f
        def encode(img):
            return first_stage_model.encode(img)[2][2].reshape(-1, size**2)
    
        def decode(z):
            # Decode from indices
            with torch.no_grad():
                z = z.reshape(-1, size, size)
                z = first_stage_model.quantize.embedding(z).permute(0,3,1,2)
                return first_stage_model.decode(z)
        return encode, decode
    else:
        raise ValueError(f"Model {config.name} is not supported.")
    
class SEDD_latent(SEDD):
    def __init__(self, model_path, first_stage_config, device='cuda'):
        super().__init__(model_path, device)
        self.encode, self.decode = load_first_stage_model(first_stage_config, device)
        