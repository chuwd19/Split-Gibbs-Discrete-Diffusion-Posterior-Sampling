import torch
import pandas as pd
import typing
import math
import applications.drakes_dna.utils as utils
import numpy as np
import os

base_path = 'applications/data_and_model/'
LOGGER = utils.get_logger(__name__)
DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3} #, 'M': 4}
INDEX_TO_DNA = {v: k for k, v in DNA_ALPHABET.items()}
lookup_array = np.array([INDEX_TO_DNA[i] for i in range(len(INDEX_TO_DNA))])


def dna_detokenize(seq):
  return ''.join([list(DNA_ALPHABET.keys())[int(i)] for i in seq])

def batch_dna_detokenize(batch_seq):
    """
    batch_seq: numpy array of shape [batch_size, seq_len]
    return: list of strings
    """
    detokenized_batch = lookup_array[batch_seq]
    detokenized_batch = [''.join(seq) for seq in detokenized_batch]
    return detokenized_batch

def dna_tokenize(seq):
  return [DNA_ALPHABET[c] for c in seq]

def batch_dna_tokenize(batch_seq):
    """
    batch_seq: list of strings
    return: numpy array of shape [batch_size, seq_len]
    """
    tokenized_batch = np.array([[DNA_ALPHABET[c] for c in seq] for seq in batch_seq])
    return tokenized_batch
  
def gosai2sorted_array():
  df = pd.read_csv(os.path.join(base_path, 'mdlm/gosai_data/dataset.csv.gz'), index_col=0) # (735156, 5)
  array = df.sort_values('hepg2').to_numpy()[:, 1:3]
  return array
  
def gosai2array():
  df = pd.read_csv(os.path.join(base_path, 'mdlm/gosai_data/dataset.csv.gz'), index_col=0) # (735156, 5)
  array = df.to_numpy()[:, 1:3]
  return array

class GosaiDataset(torch.utils.data.Dataset):
    def __init__(self, data_path='mdlm/gosai_data/processed_data/gosai_all.csv'):
        # data_df = pd.read_csv(os.path.join(base_path, f'mdlm/gosai_data/processed_data/gosai_all.csv'))
        data_df = pd.read_csv(os.path.join(base_path, data_path))
        self.seqs = torch.tensor(data_df['seq'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
        self.clss = torch.tensor(data_df[['hepg2', 'k562', 'sknsh']].to_numpy())
        self.df = data_df
        LOGGER.info(f'Loaded data: seqs shape: {self.seqs.shape}, clss shape: {self.clss.shape}')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}
      
      
class GosaiFPODataset(torch.utils.data.Dataset):
  def __init__(self, K=25):
    self.K = K
    self.data = self._tokenize_data(gosai2array())
    self.num_batches = math.ceil(len(self.data) / K)
    
  def __len__(self):
    return self.num_batches
      
  def _tokenize_sequence(self, sequence):
    return torch.tensor([DNA_ALPHABET[base] for base in sequence])
  
  def _tokenize_data(self, data):
    tokenized_data = []
    for i in range(len(data)):
      tokenized_data.append([self._tokenize_sequence(data[i][0]), data[i][1]])
    return tokenized_data
  
  def __getitem__(self, idx):
    start = idx * self.K
    end = min((idx + 1) * self.K, len(self.data))
    
    batch = self.data[start:end]
    seqs, rewards = zip(*batch)
    
    seqs = torch.stack(seqs, dim=0)
    rewards = torch.tensor(rewards)
    
    return {'seqs': seqs, 'rewards': rewards, 'attention_mask': torch.ones(len(seqs))}
  
class GosaiDPODataset(torch.utils.data.Dataset):
  def __init__(self, get_pairs, num_pairs=30000):
    '''
    Initializes the Gosai DPO dataset. 
    
    get_pairs: function that takes as input the sorted data and the number of pairs and creates the pair dataset.
    num_pairs: the size of the DPO dataset to be created.
    '''
    self.num_pairs = num_pairs
    self.sorted_data = gosai2sorted_array()
    self.all_pairs = self._tokenize_pairs(get_pairs(self.sorted_data, num_pairs))
    
  def _tokenize_sequence(self, sequence):
    return torch.tensor([DNA_ALPHABET[base] for base in sequence])
  
  def _tokenize_pairs(self, pairs):
    tokenized_pairs = []
    for win, lose, win_hepg2, lose_hepg2 in pairs:
        tokenized_win = self._tokenize_sequence(win)
        tokenized_lose = self._tokenize_sequence(lose)
        tokenized_pairs.append([tokenized_win, tokenized_lose, win_hepg2, lose_hepg2])
    return tokenized_pairs
    
  def __len__(self):
    return len(self.all_pairs)
  
  def __getitem__(self, idx):
    return {'win': self.all_pairs[idx][0], 'lose': self.all_pairs[idx][1], 
            'win_hepg2': self.all_pairs[idx][2], 'lose_hepg2': self.all_pairs[idx][3], 
            'attention_mask': torch.ones(len(self.all_pairs[idx][0]))}


def get_datasets_gosai():
  return GosaiDataset()

def get_dpo_datasets_gosai(get_pairs, num_pairs):
  return GosaiDPODataset(get_pairs, num_pairs)


def get_dataloaders_gosai(config, skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  train_set = GosaiDataset()
  # randomly sample a subset of the train_set as valid_set
  valid_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))
  test_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=not config.data.streaming,
    persistent_workers=True)
  if skip_valid:
    valid_loader = None
    test_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)

  return train_loader, valid_loader, test_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py
class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0

