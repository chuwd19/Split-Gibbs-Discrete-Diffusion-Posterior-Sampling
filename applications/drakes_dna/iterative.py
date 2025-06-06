from utils import set_seed
from tqdm import tqdm
import lightning as L
from argparse import ArgumentParser
import os
from hydra import initialize, compose
from dpo_gosai import DiffusionFPO
from lightning import Trainer
from dataloader_gosai import GosaiDataset, GosaiFPODataset, get_dataloaders_gosai
from gen_pairs import create_contrastive_pairs
import torch
from eval import eval_model
import wandb
from gen_data import get_samples_and_rewards
import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base_path', type=str, default='data_and_model/')
    parser.add_argument('--model_path', type=str, default='mdlm/outputs_gosai/fpo_finetuned.pt')
    parser.add_argument('--ref_model_path', type=str, default='mdlm/outputs_gosai/pretrained.ckpt')
    parser.add_argument('--data_path', type=str, default='mdlm/gosai_data/processed_data/fpo_data.csv')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--K', type=int, default=2000)
    parser.add_argument('--config_path', type=str, default='configs_gosai')
    parser.add_argument('--config_name', type=str, default='config_dpo_gosai.yaml')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--save_path', type=str, default='results.pt')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--generator', type=str, default='rkl')
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--temporary_path', type=str, default='temp_fpo_data.csv')
    parser.add_argument('--sample_every', type=int, default=2)
    args = parser.parse_args()
    
    set_seed(args.seed, use_cuda=True)
    
    initialize(config_path=args.config_path, job_name="load_model")
    config = compose(config_name=args.config_name)
    
    if args.wandb:
        wandb.init(project='DNA_iterative', group='fPO', reinit=True, settings=wandb.Settings(start_method='fork'),
                   config=vars(args))
    
    model_path = os.path.join(args.base_path, args.model_path)
    ref_path = os.path.join(args.base_path, args.ref_model_path)
    
    ref_model = DiffusionFPO.load_from_checkpoint(ref_path, config=config, beta=1.0, generator=args.generator).to('cuda')
    ref_model.eval()
    
    model = DiffusionFPO.load_from_checkpoint(ref_path, config=config, beta=args.beta)
    model.set_ref_model(ref_model)
    model.load_state_dict(torch.load(model_path))
    model.train()
    
    all_detokenized_samples, generated_preds = get_samples_and_rewards(model, 1, 1000, verbose=True, mode='train')
    df = pd.DataFrame({
        'seq': all_detokenized_samples,
        'hepg2': generated_preds[:, 0],
        'k562': generated_preds[:, 1],
        'sknsh': generated_preds[:, 2]
    })
    df.to_csv(os.path.join(args.base_path,args.temporary_path), index=True)
        

    
    optim = torch.optim.AdamW(
      model.parameters(),
      lr=args.lr,
      betas=(args.beta1,
             args.beta2),
      eps=args.eps,
      weight_decay=args.weight_decay)
    
    pbar = tqdm(range(args.num_epochs))
    device = torch.device('cuda')
    
    for i in pbar:
        fpo_dataset = GosaiDataset(data_path=args.temporary_path)
        fpo_train_loader = torch.utils.data.DataLoader(
            fpo_dataset,
            batch_size=args.K,
            shuffle=True
        )
        
        
        # ckpt = model.state_dict()  # Online ref model
        # for key in list(ckpt.keys()):
        #     if "ref_model" in key:
        #         del ckpt[key]
        # ref_model.load_state_dict(ckpt)
        # ref_model.eval()
        # model.set_ref_model(ref_model)
    
        
        total_loss = 0.
        
        
        pbar2 = tqdm(enumerate(fpo_train_loader))
            
        for idx, batch in pbar2:
            batch['seqs'] = batch['seqs'].to(device)
            batch['clss'] = batch['clss'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            
            loss = model.training_step(batch, idx)
            loss.backward()
            
            if config.trainer.gradient_clip_val > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.trainer.gradient_clip_val)
                
            optim.step()
            total_loss += loss.item()
            
            pbar2.set_description(f'Batch: {idx}. Train loss: {loss.item()}')
            
        avg_loss = total_loss / len(fpo_train_loader)
        pbar.set_description(
            (
                f'Epoch: {i}. Avg. Train loss: {avg_loss}'
            )
        )
        
        # Generate samples and save
        if (i + 1) % args.sample_every == 0:
            all_detokenized_samples, generated_preds = get_samples_and_rewards(model, 1, 1000, verbose=True,mode='train')
        
            df = pd.DataFrame({
                'seq': all_detokenized_samples,
                'hepg2': generated_preds[:, 0],
                'k562': generated_preds[:, 1],
                'sknsh': generated_preds[:, 2]
            })
            df.to_csv(os.path.join(args.base_path,args.temporary_path), index=True)
            
        
        # if wandb.run is not None:
        #     wandb.log({'training_steps/pred_hepg2': generated_preds[:, 0].mean()})
        #     wandb.log({'training_steps/log_lik': model_logl.mean()})
        #     wandb.log({'training_steps/atac': generated_atac_acc})
        #     wandb.log({'training_steps/p_coef': generated_p_coef})
        if (i + 1) % args.sample_every == 0:
            model.eval()
            all_detokenized_samples, model_logl, generated_preds, generated_atac_acc, generated_p_coef = eval_model(model, ref_model, 10, 64, args.verbose)
            wandb.log({'pred_hepg2': generated_preds[:, 0].mean()})
            wandb.log({'log_lik': model_logl.mean()})
            wandb.log({'atac': generated_atac_acc})
            wandb.log({'p_coef': generated_p_coef})
            model.train()
        
        # if (i + 1) % args.eval_every == 0:
        #     model.eval()
        #     all_detokenized_samples, model_logl, generated_preds, generated_atac_acc, generated_p_coef = eval_model(model, ref_model, 10, 64, args.verbose)
        #     if wandb.run is not None:
        #         wandb.log({'pred_hepg2': generated_preds[:, 0].mean()})
        #         wandb.log({'log_lik': model_logl.mean()})
        #         wandb.log({'atac': generated_atac_acc})
        #         wandb.log({'p_coef': generated_p_coef})
        #     model.train()
        
    model.eval()
    all_detokenized_samples, model_logl, generated_preds, generated_atac_acc, generated_p_coef = eval_model(model, ref_model, 10, 64, args.verbose)
    
    torch.save(model.state_dict(), args.save_path)
    
    if wandb.run is not None:
        wandb.log({'pred_hepg2': generated_preds[:, 0].mean()})
        wandb.log({'log_lik': model_logl.mean()})
        wandb.log({'atac': generated_atac_acc})
        wandb.log({'p_coef': generated_p_coef})
        wandb.log({'reward_atac': generated_preds[:, 0].mean() * generated_atac_acc})
        wandb.log({'total_metric': generated_preds[:, 0].mean() * generated_atac_acc * generated_p_coef})
        wandb.log({'reward_pearson': generated_preds[:, 0].mean() * generated_p_coef})
