import os
from omegaconf import OmegaConf
import pickle
import hydra
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
import wandb

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # save config 
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))
    net = instantiate(config.model.model)
    forward_model = instantiate(config.problem.model, device=device)
    if config.measurement:
        dataset = instantiate(config.problem.data)
        dataLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    algorithm = instantiate(config.algorithm.method, net=net, forward_op=forward_model)
    evaluator = instantiate(config.problem.evaluator, device=device)
    visualizer = instantiate(config.problem.visualizer, save_dir=exp_dir)
    
    if config.wandb:
        wandb.init(project="discrete-guidance")
    gt, pred, gt_y, pred_y = [], [], [], []
    
    if config.measurement:
        # Inverse problem setting
        for i, data in enumerate(dataLoader):
            data = data.to(device)
            y = forward_model(data)
            samples = algorithm.inference(observation=y, num_samples=config.batch_size)
            save_path = os.path.join(exp_dir, f'result_{i}.pt')
            torch.save(samples, save_path)
            evaluator(data, samples, y, forward_model(samples))
            stats = evaluator.compute()
            if config.verbose:
                print(stats)
            
            gt.append(data)
            pred.append(samples)
            gt_y.append(y)
            pred_y.append(forward_model(samples))
            
    else:
        # Reward function setting
        # TODO: batch this to speed it up?
        rounds = config.num_samples // config.batch_size
        for i in range(rounds):
            samples = algorithm.inference(observation=torch.zeros(config.batch_size), num_samples=config.batch_size)
            save_path = os.path.join(exp_dir, f'result_{i}.pt')
            torch.save(samples, save_path)
            evaluator(samples, forward_model(samples))
            stats = evaluator.compute()
            if config.verbose:
                print(stats)
            
    stats = evaluator.compute()
    if config.wandb:
        wandb.log(stats)
    print(stats)
    
    ## Visualize
    if config.measurement:
        gt = torch.cat(gt, dim=0)
        pred = torch.cat(pred, dim=0)
        gt_y = torch.cat(gt_y, dim=0)
        pred_y = torch.cat(pred_y, dim=0)
        visualizer(gt, pred, gt_y, pred_y)

if __name__ == "__main__":
    main()