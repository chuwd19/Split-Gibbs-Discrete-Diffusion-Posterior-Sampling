from .base import BaseOperator
import torch
from models.mnist import SimpleConv

class MNIST_classifier(BaseOperator):
    def __init__(self, device='cuda', **kwargs):
        super().__init__()
        self.classifier = SimpleConv()
        self.classifier.load_state_dict(torch.load("checkpoints/mnist_classifier.pt"))
        self.classifier.to(device)
        
    def _decompose(self, x):
        scale = 256 / 2
        return (x * scale).float().reshape(-1, 1, 32, 32)
    
    def __call__(self, inputs):
        inputs = self._decompose(inputs)
        return self.classifier(inputs)
    
    def loss(self, inputs, y):
        return torch.nn.functional.cross_entropy(self(inputs), torch.zeros(inputs.shape[0], device=inputs.device).long(), reduction='none')
    
    
    