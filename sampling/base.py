from abc import ABC, abstractmethod


class Algo(ABC):
    '''
    net: Discrete Diffusion Model
    forward_op: Forward Operator
    '''
    def __init__(self, net, forward_op):
        self.net = net
        self.forward_op = forward_op
    
    @abstractmethod
    def inference(self, observation=None, num_samples=1, **kwargs):
        '''
        Args:
            - observation: observation for one single ground truth
            - num_samples: number of samples to generate for each observation
        '''
        pass