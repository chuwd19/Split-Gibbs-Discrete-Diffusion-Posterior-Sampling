from utils import set_seed
from argparse import ArgumentParser
import os
from hydra import initialize, compose
from dpo_gosai import DiffusionFPO
import torch
from tqdm import tqdm
import dataloader_gosai
import numpy as np
import oracle
import pandas as pd

def get_samples_and_rewards(model, num_sample_batches, num_samples_per_batch, verbose=True, mode='train'):
    all_detokenized_samples = []
    all_raw_samples = []
    for _ in tqdm(range(num_sample_batches)):
        samples = model._sample(eval_sp_size=num_samples_per_batch)
        all_raw_samples.append(samples)
        detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
        all_detokenized_samples.extend(detokenized_samples)
    all_raw_samples = torch.concat(all_raw_samples)
            
    generated_preds = []
    for i in range(num_sample_batches):
        start = i * num_samples_per_batch
        end = (i + 1) * num_samples_per_batch
        generated_preds.append(oracle.cal_gosai_pred_new(all_detokenized_samples[start : end], mode=mode))
    generated_preds = np.concatenate(generated_preds, axis=0)
    
    if verbose:
        print(f'Model avg. generated sample predicted HepG2: {generated_preds[:, 0].mean()}, median predicted HepG2: {np.median(generated_preds[:, 0])}')
    
    return all_detokenized_samples, generated_preds

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base_path', type=str, default='data_and_model/')
    parser.add_argument('--model_path', type=str, default='mdlm/outputs_gosai/fpo_10epochs.pt')
    parser.add_argument('--ref_model_path', type=str, default='mdlm/outputs_gosai/pretrained.ckpt')
    parser.add_argument('--config_path', type=str, default='configs_gosai')
    parser.add_argument('--config_name', type=str, default='config_dpo_gosai.yaml')
    parser.add_argument('--save_path', type=str, default='fpo_data.csv')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--num_sample_batches', type=int, default=100)
    parser.add_argument('--num_samples_per_batch', type=int, default=1000)
    args = parser.parse_args()
    
    set_seed(args.seed, use_cuda=True)
    
    initialize(config_path=args.config_path, job_name="load_model")
    config = compose(config_name=args.config_name)
    
    model_path = os.path.join(args.base_path, args.model_path)
    ref_path = os.path.join(args.base_path, args.ref_model_path)
    
    ref_model = DiffusionFPO.load_from_checkpoint(ref_path, config=config, beta=1.0).to('cuda')
    ref_model.eval()
        
    model = DiffusionFPO.load_from_checkpoint(ref_path, config=config, beta=1.0).to('cuda')
    model.set_ref_model(ref_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_detokenized_samples, generated_preds = get_samples_and_rewards(model, args.num_sample_batches, args.num_samples_per_batch, verbose=True, mode='train')
        
    df = pd.DataFrame({
        'seq': all_detokenized_samples,
        'hepg2': generated_preds[:, 0],
        'k562': generated_preds[:, 1],
        'sknsh': generated_preds[:, 2]
    })
    df.to_csv(args.save_path, index=True)