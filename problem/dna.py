
from torch.nn import functional as F
from .base import BaseOperator
from grelu.lightning import LightningModel
from applications.drakes_dna.oracle import get_gosai_oracle

class DNA(BaseOperator):
    def __init__(self, model, sigma_noise=0.01, length = 200, device='cuda'):
        super().__init__(sigma_noise, device)
        self.length = length
        # reward_model = LightningModel.load_from_checkpoint(model, map_location='cpu').to(device)
        reward_model = get_gosai_oracle(mode='train').to(device)
        reward_model.eval()
        self.model = reward_model

    
    def transform_samples(self, samples, num_classes=4):
        # One-hot encode the tensor but first mask out the '4's
        mask = samples != 4
        # CGAT -> ACGT
        samples_t = samples.clone()
        samples_t[samples == 0] = 1
        samples_t[samples == 1] = 2
        samples_t[samples == 2] = 0
        
        valid_samples = (samples_t * mask)
        one_hot_samples = F.one_hot(valid_samples, num_classes=num_classes)

        # Apply mask to zero out invalid rows
        one_hot_samples = one_hot_samples * mask.unsqueeze(-1)
        return one_hot_samples


    def __call__(self, inputs):
        return self.model(self.transform_samples(inputs).float().transpose(1,2))[:, 0]
    
    def loss(self, inputs, y):
        return -((self(inputs))).flatten(1).sum(-1)
    
    def log_likelihood(self, inputs, y=None, **kwargs):
        return -self.loss(inputs, y, **kwargs)/self.sigma_noise