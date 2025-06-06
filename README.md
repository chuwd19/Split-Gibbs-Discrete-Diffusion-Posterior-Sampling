# SGDD: Split Gibbs Discrete Diffusion Posterior Sampling

## Introduction

We propose a principled plug-and-play discrete diffusion sampling method, called **S**plit **G**ibbs **D**iscrete **D**iffusion Posterior Sampling. Our method solves inverse problems and generate reward guided samples in discrete-state spaces using discrete diffusion models as a prior.




## Local Setup

### Prepare the environment


- python 3.9
- PyTorch 2.3  
- CUDA 11.8

Other versions of PyTorch with proper CUDA should work but are not fully tested.

```bash
conda create -n SGDD python=3.9
conda activate SGDD

pip install -r requirements.txt
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Posterior sampling with SGDD

To run **DNA design** with `SGDD`:

```
python main.py problem=dna model=dna algorithm=sgdd measurement=False num_samples=640 batch_size=10

```

The results are saved at foloder `exps`.

