from evodiff.pretrained import D3PM_UNIFORM_38M
from .base import DiscreteDiffusion
import torch.nn.functional as F
import torch
import tqdm
from typing import Optional, Callable, Literal


class D3PM(DiscreteDiffusion):
    def __init__(self, model_name, seq_len=100, device='cuda'):
        if model_name == 'd3pm_uniform_38m':
            model, _, tokenizer, scheme, timestep, Q_bar, Q = D3PM_UNIFORM_38M(return_all=True)
            self.S = tokenizer.K # -6 # remove nonstandard amino acids (?)
            self.model = model
            self.model.eval()
            self.model.to(device)
            self.tokenizer = tokenizer
            self.scheme = scheme
            self.timestep = timestep
            self.Q_bar = Q_bar.to(device)[:, :self.S, :self.S]
            self.Q = Q.to(device)[:, :self.S, :self.S]
        else:
            raise ValueError(f"Model {model_name} is not valid.")

        self.device = device
        self.length = seq_len

    def score(self, x, t):
        pass
    
    def pred_mean(self, x, t):
        # Predict a distribution of p(x_0)
        t = torch.tensor([t]*x.shape[0]).to(self.device)
        return F.softmax(self.model(x,t)[:,:,:self.S], dim=-1)

    def get_start(self, batch_size):
        return torch.randint(0, self.S, (batch_size, self.length)).to(torch.long).to(self.device)
    
    def q_sample(self, x, t):
        # forward process
        # x: (batch_size, seq_len)
        prob = self.Q_bar[t][x] # (batch_size, seq_len, K)
        return torch.multinomial(prob.view(-1, prob.shape[-1]), num_samples=1).view(prob.shape[:-1])
    
    ### Modified from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py. ### 
    def get_all_jump_transitions(
        self,
        x: torch.Tensor,  # Shape: (B, D)
    ) -> torch.Tensor:  # Shape: (B*D*S, D)
        """
        Gets all possible single-dimension transitions from current states.

        Creates a tensor containing all possible states that differ from input states
        in exactly one position, for each sequence in the batch.

        Args:
            xt: Current state tensor of shape (batch_size, sequence_length)
            S: Size of categorical state space (number of possible values per position)

        Returns:
            Tensor of shape (batch_size * sequence_length * state_space, sequence_length)
            containing all possible single-token transitions
        """
        B, D = x.shape
        device = x.device

        # Create B*D*S copies of input sequences
        # Shape: (B, 1, D) -> (B, D*S, D)
        xt_expand = x.unsqueeze(1).repeat(1, D * self.S, 1)
        # Flatten batch and transition dimensions
        # Shape: (B, D*S, D) -> (B*D*S, D)
        xt_expand = xt_expand.view(-1, D)

        # Create indices for all possible transitions
        # Shape: (D*S,) -> (B, D*S) -> (B*D*S,)
        jump_idx = torch.arange(D * self.S).to(device)
        jump_idx = jump_idx.repeat(B, 1).flatten()

        # Create tensor for states after one transition
        xt_jumps = xt_expand.clone()

        # Calculate which dimension changes for each transition
        # Shape: (B*D*S,)
        jump_dims = jump_idx // self.S

        # Calculate new value for changed dimension
        # Shape: (B*D*S,)
        jump_states = jump_idx % self.S

        # Apply transitions by assigning new values at transition dimensions
        # Shape: (B*D*S, D)
        xt_jumps[
            torch.arange(jump_idx.size(0), device=device),
            jump_dims,  # Index the transitioned dimension
        ] = jump_states  # Assign the new state

        return xt_jumps
    
    ## Modified from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py. ### 
    def get_guided_rates(
        self,
        predictor_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,  # Shape: (B, D)
        t: float,
        q_t: torch.Tensor,  # Shape: (B, D, S)
        use_tag: bool = False,
        guide_temp: float = 1.0,
        log_prob_ratio_cutoff: float = 80.0,
    ) -> torch.Tensor:
        """
        Computes guide-adjusted rates for predictor guidance.

        Implements both exact guidance by computing likelihood ratios for all possible transitions,
        and Taylor-approximated guidance (TAG) using gradients of the predictor.

        Args:
            predictor_log_prob (callable): Function that takes (x,t) and returns log p(y|x,t)
            xt (torch.Tensor): Current states of shape (B, D)
            t (float): Current time
            R_t (torch.Tensor): Unconditional rates of shape (B, D, S)
            S (int): Size of categorical state space
            use_tag (bool, optional): Whether to use Taylor approximation. Defaults to False.
            guide_temp (float, optional): Guidance temperature. Defaults to 1.
            log_prob_ratio_cutoff (float, optional): Maximum value for log ratios. Defaults to 80.

        Returns:
            torch.Tensor: Guide-adjusted rates of shape (B, D, S)
        """
        B, D = x.shape
        device = x.device
        t = t * torch.ones((B,), device=device)
        if not use_tag:
            # Exact guidance case
            # log p(y|x=z_t), shape (B,)
            log_prob_xt = predictor_model(x, t)

            # Get all jump transitions, shape (B*D*S, D)
            xt_jumps = self.get_all_jump_transitions(x, self.S)

            # Get log probs for all transitions
            # Shape: (B*D*S,) -> (B, D, S)
            log_prob_xt_jumps = predictor_model(
                xt_jumps, t.repeat(1, D * self.S).flatten()
            ).view(B, D, self.S)

            # Compute log ratios
            # Shape (B, D, S)
            log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)

        else:
            # Taylor-approximated guidance (TAG) case
            # One-hot encode categorical data, shape (B, D, S)
            xt_ohe = F.one_hot(x.long(), num_classes=self.S).to(torch.float)

            # \grad_{x}{log p(y|x)}(z_t), shape (B, D, S)
            with torch.enable_grad():
                xt_ohe.requires_grad_(True)
                # log p(y|x=z_t), shape (B,)
                log_prob_xt_ohe = predictor_model(xt_ohe, t)
                log_prob_xt_ohe.sum().backward()
                # Shape (B, D, S)
                grad_log_prob_xt_ohe = xt_ohe.grad
            # 1st order Taylor approximation of the log difference
            # Shape (B, D, S)
            log_prob_ratio = grad_log_prob_xt_ohe - (xt_ohe * grad_log_prob_xt_ohe).sum(
                dim=-1, keepdim=True
            )

        # Scale log prob ratio by temperature
        log_prob_ratio /= guide_temp

        # Clamp the log prob ratio to avoid overflow in exp
        log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff)
        # Exponentiate to get p(y|x=z~) / p(y|x=z_t)
        prob_ratio = torch.exp(log_prob_ratio)
        # Multiply the reverse rate elementwise with the density ratio
        # Note this doesn't deal with the diagonals
        q_t = q_t * prob_ratio

        if q_t.isnan().any():
            raise ValueError(f"The rate matrix 'q_t' contains NaNs.")

        return q_t

    @torch.no_grad()
    def p_sample(self, x, t, predictor_model=None, hard=True):
        """
        Runs a single forward step of the diffusion process.
        TODO: add argument for guided, conditional sampling.
        """
        p0 = self.pred_mean(x, t).to(torch.float64)
        x_next = []
        for i, s in enumerate(x):
            A = self.Q[t].T [s]
            Q_expand = self.Q_bar[t-1].expand(self.length, self.S, self.S)
            B_pred = torch.mul(p0[i].unsqueeze(2), Q_expand)

            #TODO: option to update rate matrix to be the guided adjusted rate matrix, add other kwargs here
            if predictor_model is not None:
                q_t = self.get_guided_rates(predictor_model, x, t, q_t)
             
            q_t = torch.mul(A.unsqueeze(1), B_pred)
            p_theta_marg = torch.bmm(torch.transpose(q_t, 1,2), p0[i].unsqueeze(2)).squeeze() 
            p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
            if hard:
                x_next.append(torch.multinomial(p_theta_marg, num_samples=1).squeeze())
            else:
                x_next.append(p_theta_marg)
        return torch.stack(x_next, dim=0).to(self.device)

    @torch.no_grad()
    def uncond_sample(self, batch_size=1, verbose=False, detokenize=True):
        """
        Sampling from the diffusion model.
        Performs unconditional sampling if predictor_model is None. Otherwise, performs guided sampling.
        """
        timesteps = torch.linspace(self.timestep-1,1,int((self.timestep-1)/1), dtype=int) # iterate over reverse timesteps
        timesteps = tqdm.tqdm(timesteps) if verbose else timesteps
        x = self.get_start(batch_size)
        for t in timesteps:
            x = self.p_sample(x, t)
        if detokenize:
            detokenized = [self.tokenizer.untokenize(s) for s in x]
            return detokenized
        else:
            return x
