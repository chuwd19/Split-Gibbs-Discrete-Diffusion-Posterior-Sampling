import os
# import diffusion_gosai_update
from dpo_gosai import *
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from dataloader_gosai import *
import torch
from tqdm import tqdm
from torch.optim import Adam
from eval import eval_model
from utils import set_seed


def create_contrastive_pairs(sorted_data, num_pairs):
    all_pairs = []
    for i in range(num_pairs):
        win = sorted_data[-1 - i]
        lose = sorted_data[i]
        pair = [win[0], lose[0], win[1], lose[1]]
        all_pairs.append(pair)
    return all_pairs

# if __name__ == '__main__':
    
#     set_seed(0, use_cuda=True)
    
#     base_path = 'data_and_model/'

#     dpo_dataset = GosaiDPODataset(create_contrastive_pairs, num_pairs=20000)
#     dpo_train_loader = torch.utils.data.DataLoader(
#         dpo_dataset,
#         batch_size=128,
#         shuffle=False
#     )
    
#     GlobalHydra.instance().clear()
    
#     config_path='configs_gosai'
#     config_name='config_dpo_gosai.yaml'

#     initialize(config_path=config_path, job_name="load_model")
#     config = compose(config_name=config_name)
#     model_path = os.path.join(base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
    
#     ref_model = DiffusionDPO.load_from_checkpoint(model_path, config=config, beta=1.0).to('cuda')
#     ref_model.eval()
    
#     # model = DiffusionDPO.load_from_checkpoint(model_path, config=config, beta=1.0).to('cuda')
#     model = DiffusionDPO.load_from_checkpoint(model_path, config=config, beta=5000.)
#     model.train()
    
#     model.set_ref_model(ref_model)
    
#     # optimizer = Adam(model.parameters(), lr=config.optim.lr)
#     optimizer = Adam(model.parameters(), lr=5e-6)
#     pbar = tqdm(range(6))
    
#     device = torch.device('cuda')
    
#     print(f'Beta: {model.beta}, LR: {optimizer.param_groups[0]['lr']}, num_pairs: {dpo_dataset.num_pairs}')
    
#     for i in pbar:
#         total_loss = 0.
#         for idx, batch in enumerate(dpo_train_loader):
#             batch['win'] = batch['win'].to(device)
#             batch['lose'] = batch['lose'].to(device)
#             batch['attention_mask'] = batch['attention_mask'].to(device)
#             loss = model.training_step(batch, idx)
            
#             loss.backward()
            
#             if config.trainer.gradient_clip_val > 0.:
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), max_norm=config.trainer.gradient_clip_val)
                
#             optimizer.step()
#             total_loss += loss.item()
        
#         avg_loss = total_loss / len(dpo_train_loader)
#         pbar.set_description(
#             (
#                 f'Epoch: {i}. Train loss: {avg_loss}'
#             )
#         )
        
#     model.eval()
#     eval_model(model, ref_model, 10, 64)
    
#     # callbacks = []
#     # if 'callbacks' in config:
#     #     for _, callback in config.callbacks.items():
#     #         callbacks.append(hydra.utils.instantiate(callback))
    
#     # trainer = hydra.utils.instantiate(
#     #     config.trainer,
#     #     default_root_dir=os.getcwd(),
#     #     strategy=hydra.utils.instantiate(config.strategy),
#     #     )    
#     # print('Starting training')
#     # trainer.fit(model, dpo_train_loader)
    
#     # for idx, batch in enumerate(dpo_train_loader):
#     #     batch['win'] = batch['win'].to('cuda')
#     #     batch['lose'] = batch['lose'].to('cuda')
#     #     batch['attention_mask'] = batch['attention_mask'].to('cuda')
#     #     total_loss = model.training_step(batch, idx)
#     #     print(total_loss)
#     #     # win_seqs = batch['win'].to('cuda')
#     #     # lose_seqs = batch['lose'].to('cuda')
#     #     # attn_mask = batch['attention_mask'].to('cuda')
#     #     # wl = model._compute_loss({'seqs': win_seqs, 'attention_mask': attn_mask}, 'train')
#     #     # ll = model._compute_loss({'seqs': lose_seqs, 'attention_mask': attn_mask}, 'train')
#     #     if idx >= 10:
#     #         break
        