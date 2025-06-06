import torch
from abc import ABC, abstractmethod
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import os

class Visualizer(ABC):
    def __init__(self, save_dir=None):
        pass

    def __call__(self, gt, pred, gt_y, pred_y):
        pass


class MNIST_visualizer(Visualizer):
    def __init__(self, decompose=True, save_dir=None):
        self.decompose = decompose
        self.save_dir = save_dir
        
    def _decompose(self, x):
        scale = 256 / 2
        return (x * scale).float().reshape(-1, 1, 32, 32)
    
    def __call__(self, gt, pred, gt_y, pred_y):
        if self.decompose:
            gt = self._decompose(gt)
            pred = self._decompose(pred)
        if self.save_dir:
            images = make_grid(torch.cat([gt, pred], dim=0), nrow = gt.shape[0], padding=4, pad_value=1)
            save_image(images, os.path.join(self.save_dir, "results.png"))
            
            
class FFHQ_visualizer(Visualizer):
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        
    
    def __call__(self, gt, pred, gt_y, pred_y):
        if self.save_dir:
            images = make_grid(torch.cat([gt, pred], dim=0), nrow = gt.shape[0], padding=4, pad_value=1)
            save_image(images*0.5+0.5, os.path.join(self.save_dir, "results.png"))
            
            
class Gaussian_visualizer(Visualizer):
    def __init__(self, grid_solution, save_dir=None):
        self.save_dir = save_dir
        self.grid_solution = grid_solution
    
    def statistics2d(self, x, grid_solution=25):
        map = torch.zeros(grid_solution, grid_solution)
        for i in range(len(x)):
            map[x[i,0], x[i,1]] += 1
        return map/len(x)
 
    def __call__(self, gt, pred, gt_y, pred_y):
        heatmap = self.statistics2d(pred, self.grid_solution)
        if self.save_dir:
            save_path = os.path.join(self.save_dir, "results.png")
            plt.figure(figsize=(5,5), dpi=100)
            plt.imshow(heatmap, cmap='hot')
            plt.savefig(save_path)
            plt.tight_layout()
            plt.close()