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

### Pretrained Models

Please download pretrained discrete diffusion prior weights in the GitHub release page. The models are trained using the [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) codebase. To run the default code, put the pretrained models under the `checkpoints` folder.

#### Reward Oracles for DNA Design

We use the reward oracles used for DNA design from [DRAKES](https://github.com/ChenyuWang-Monica/DRAKES). Please download the data and model weights in their original [release](https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=0) and put them in the `applications` folder (their original page [here](https://github.com/ChenyuWang-Monica/DRAKES/tree/master/drakes_dna)).

### Posterior sampling with SGDD

Example: to run **DNA design** with `SGDD`:

```
python main.py problem=dna model=dna algorithm=sgdd measurement=False num_samples=640 batch_size=10
```

The results are saved at the folder `exps`.

