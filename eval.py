import torch
from abc import ABC, abstractmethod
from piq import psnr, LPIPS
import torch.nn as nn
from problem.basic import SingleModal
from models.mnist import SimpleConv
import numpy as np
from applications.drakes_dna.oracle import get_gosai_oracle, cal_atac_pred_new, count_kmers, cal_highexp_kmers
import torch.nn.functional as F
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from applications.drakes_dna.diffusion_gosai_update import Diffusion
from applications.drakes_dna.dataloader_gosai import batch_dna_detokenize
from scipy.stats import pearsonr
from grelu.interpret.motifs import scan_sequences
import pandas as pd

class Evaluator(ABC):
    def __init__(self, device="cuda"):
        self.metric_list = {}
        self.metric_state = {key: 0.0 for key in self.metric_list.keys()}

    def __call__(self, gt, pred, gt_y, pred_y):
        pass
    
    def compute(self):
        return 0
    
    def reset(self):
        self.metric_state = {key: 0.0 for key in self.metric_list.keys()}


class MNIST_evaluator_cls(Evaluator):
    def __init__(self, device="cuda", decompose=True):
        self.classifier = SimpleConv()
        self.classifier.load_state_dict(torch.load("checkpoints/mnist_model.pth"))
        self.classifier.to(device)
        self.metric_list = {
            'Acc': lambda gt, pred, gt_y, pred_y: (self.classifier(pred).argmax(dim=1) == torch.zeros(pred.shape[0],device=pred.device)).float(),
        }
        self.metric_state = {key: [] for key in self.metric_list.keys()}
        self.decompose = decompose
        
    def _decompose(self, x):
        scale = 256 / 2
        return (x * scale).float().reshape(-1, 1, 32, 32)
    
    def __call__(self, gt, pred, gt_y, pred_y):
        if self.decompose:
            gt = self._decompose(gt)
            pred = self._decompose(pred)
        for key, metric in self.metric_list.items():
            self.metric_state[key].append(metric(gt, pred, gt_y, pred_y))
        
    def compute(self):
        return {key: torch.cat(value).mean().item() for key, value in self.metric_state.items()}
    



class MNIST_evaluator(Evaluator):
    def __init__(self, device="cuda", decompose=True):
        self.classifier = SimpleConv()
        self.classifier.load_state_dict(torch.load("checkpoints/mnist_model.pth"))
        self.classifier.to(device)
        self.metric_list = {
            'PSNR': lambda gt, pred, gt_y, pred_y: psnr(gt.clip(0, 1), pred.clip(0, 1), data_range=1.0, reduction='none'),
            'Acc': lambda gt, pred, gt_y, pred_y: (self.classifier(gt).argmax(dim=1) == self.classifier(pred).argmax(dim=1)).float(),
            'Meas_fit': lambda gt, pred, gt_y, pred_y: (gt_y - pred_y).abs().float() 
        }
        self.metric_state = {key: [] for key in self.metric_list.keys()}
        self.decompose = decompose
        
    def _decompose(self, x):
        scale = 256 / 2
        return (x * scale).float().reshape(-1, 1, 32, 32)
    
    def __call__(self, gt, pred, gt_y, pred_y):
        if self.decompose:
            gt = self._decompose(gt)
            pred = self._decompose(pred)
        for key, metric in self.metric_list.items():
            self.metric_state[key].append(metric(gt, pred, gt_y, pred_y))
        
    def compute(self):
        return {key: torch.cat(value).mean().item() for key, value in self.metric_state.items()}
    

class FFHQ_evaluator(Evaluator):
    def __init__(self, device="cuda", decompose=True):
        self.metric_list = {
            'PSNR': lambda gt, pred, gt_y, pred_y: psnr((gt*0.5+0.5).clip(0, 1), (pred*0.5+0.5).clip(0, 1), data_range=1.0, reduction='none'),
            'Meas_fit': lambda gt, pred, gt_y, pred_y: (gt_y - pred_y).abs().float(),
            'LPIPS': lambda gt, pred, gt_y, pred_y: LPIPS(replace_pooling=True, reduction='none')((gt*0.5+0.5).clip(0, 1), (pred*0.5+0.5).clip(0, 1)),
        }
        self.metric_state = {key: [] for key in self.metric_list.keys()}
        self.decompose = decompose
        
    
    def __call__(self, gt, pred, gt_y, pred_y):
        for key, metric in self.metric_list.items():
            self.metric_state[key].append(metric(gt, pred, gt_y, pred_y))
        
    def compute(self):
        return {key: torch.cat(value).mean().item() for key, value in self.metric_state.items()}



def compare_kmer(kmer1, kmer2, n_sp1, n_sp2):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1:
            counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2:
            counts[i][0] = kmer2[kmer]
    return pearsonr(counts[:, 0], counts[:, 1])[0]

class DNA_evaluator(Evaluator):
    def __init__(self, device="cuda", decompose=True):
        self.decompose = decompose
        self.device = device
        self.oracle = get_gosai_oracle(mode='eval').to(device)
        self.oracle.eval()
        
        if not GlobalHydra.instance().is_initialized():
            initialize(config_path="configs") 
        config = compose(config_name="config_gosai")
        self.model = Diffusion.load_from_checkpoint('applications/data_and_model/mdlm/outputs_gosai/pretrained.ckpt', config=config)
        self.model.eval()
        self.highexp_kmers_999, self.n_highexp_kmers_999, self.highexp_seqs_999 = None, None, None
        self.metric_list = {
            'reward': lambda x, y: self.oracle(F.one_hot(x, num_classes=4).float().transpose(1, 2)).squeeze(-1)[:, 0].cpu().detach().numpy(),
            'LogLikelihood':  lambda x, y: self.model.get_likelihood(x, num_steps=128, n_samples=1).cpu().detach().numpy(),
            'ATAC': lambda x, y: (cal_atac_pred_new(x)[:,1] > 0.5),
        }
        self.metric_state = {key: [] for key in self.metric_list.keys()}

    def kmer_eval(self, x):
        if self.highexp_kmers_999 is None:
            _,_,self.highexp_kmers_999, self.n_highexp_kmers_999,_,_,self.highexp_seqs_999 = cal_highexp_kmers(return_clss=True)
        x = self.transform_samples(x)
        x = batch_dna_detokenize(x.detach().cpu().numpy())
        generated_kmer = count_kmers(x)
        generated_p_coef = compare_kmer(self.highexp_kmers_999, generated_kmer, self.n_highexp_kmers_999, len(x))
        return generated_p_coef
    
    def jaspar_eval(self, x):
        x = self.transform_samples(x)
        x = batch_dna_detokenize(x.detach().cpu().numpy())
        if self.highexp_seqs_999 is None:
            _,_,self.highexp_kmers_999, self.n_highexp_kmers_999,_,_,self.highexp_seqs_999 = cal_highexp_kmers(return_clss=True)
        motif_count = scan_sequences(x, 'jaspar')
        motif_count_sum = motif_count['motif'].value_counts()
        motif_count_top = scan_sequences(self.highexp_seqs_999, 'jaspar')
        motif_count_top_sum = motif_count_top['motif'].value_counts()
        
        motifs_summary = pd.concat([motif_count_top_sum, motif_count_sum], axis=1)
        motifs_summary.columns = ['top_data', 'finetuned']
        jaspar_corrs = motifs_summary.corr(method='spearman')
        return jaspar_corrs
    
    def transform_samples(self, samples):
        # One-hot encode the tensor but first mask out the '4's
        # CGAT -> ACGT
        samples_t = samples.clone()
        samples_t[samples == 0] = 1
        samples_t[samples == 1] = 2
        samples_t[samples == 2] = 0
        return samples_t
    
    def __call__(self, pred, pred_y):
        pred = self.transform_samples(pred)
        for key, metric in self.metric_list.items():
            self.metric_state[key].extend(metric(pred, pred_y))
        
    def compute(self):
        metric_state = {}
        for key, val in self.metric_state.items():
            metric_state[key] = np.mean(val).item()
            metric_state[f'{key}_std'] = np.std(val).item()
        return metric_state
    
    def reset(self):
        self.metric_state = {key: [] for key in self.metric_list.keys()}
    
class Pianoroll_evaluator(Evaluator):
    def __init__(self, dim=129, device="cuda", decompose=True):
        self.dim = dim
        self.metric_list = {
            'hellinger': lambda gt, pred, gt_y, pred_y: torch.sqrt(1 - (torch.sqrt(self.statistics(gt)) * torch.sqrt(self.statistics(pred))).sum(dim=-1)),
            'Meas_fit': lambda gt, pred, gt_y, pred_y: (gt_y != pred_y).sum(-1).float() 
        }
        self.metric_state = {key: [] for key in self.metric_list.keys()}
        self.decompose = decompose
    
    def statistics(self, x):
        histogram = torch.zeros(x.shape[0],self.dim)
        for i in range(len(x)):
            histogram[i] = torch.bincount(x[i], minlength=self.dim)
        return histogram/x.shape[1]
    
    def __call__(self, gt, pred, gt_y, pred_y):
        for key, metric in self.metric_list.items():
            self.metric_state[key].append(metric(gt, pred, gt_y, pred_y))
        
    def compute(self):
        metric = {key: torch.cat(value).mean().item() for key, value in self.metric_state.items()}
        metric_std = {f'{key}_std': torch.cat(value).std().item() for key, value in self.metric_state.items()}
        return metric | metric_std
        # return {key: torch.cat(value) for key, value in self.metric_state.items()}
    
class Gaussian_evaluator(Evaluator):
    def __init__(self, device="cuda", grid_solution=50, std=2):
        self.grid_solution = grid_solution
        distribution = torch.distributions.Normal(0, std)
        grid = torch.linspace(-5, 5, grid_solution)
        distribution = distribution.log_prob(grid).exp()
        distribution_2d = distribution[None, :] * distribution[:, None]
        grid_2d = torch.stack(torch.meshgrid(grid, grid), dim=-1).reshape(-1, 2)
        forward_op = SingleModal(length=2, grid_solution=grid_solution, device='cpu',sigma_noise=3)
        y = forward_op(self.discretize(torch.ones(2) * 5/3))
        score = forward_op.log_likelihood(self.discretize(grid_2d),y).reshape(grid_solution,grid_solution)
        self.gt_distribution = score.exp()*distribution_2d
        self.gt_distribution /= self.gt_distribution.sum()
        self.metric_list = {
            'hellinger': lambda emp_dist: torch.sqrt(1 - (torch.sqrt(emp_dist) * torch.sqrt(self.gt_distribution)).sum()).item(),
            'total_variance': lambda emp_dist:  (emp_dist - self.gt_distribution).abs().sum().item()/2
        }
        self.metric_state = {key: [] for key in self.metric_list.keys()}
    
    def statistics2d(self, x, grid_solution):
        map = torch.zeros(grid_solution, grid_solution)
        for i in range(len(x)):
            map[x[i,0], x[i,1]] += 1
        return map/len(x)
    
    def discretize(self, x):
        return ((x + 5)/10 * self.grid_solution).int()
    
    def __call__(self, gt, pred, gt_y, pred_y):
        emp_dist = self.statistics2d(pred, self.grid_solution)
        for key, metric in self.metric_list.items():
            self.metric_state[key].append(metric(emp_dist))
        
    def compute(self):
        return {key: torch.tensor(value).mean().item() for key, value in self.metric_state.items()}
    
