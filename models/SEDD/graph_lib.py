import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    

def get_graph(config, device):
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    elif config.graph.type == "combined":
        return Combined(config.tokens, config.graph.uniform_rate, config.graph.absorb_rate)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim
    
    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    

class Combined(Graph):

    def __init__(self, dim, uniform_rate=1, absorb_rate=1):
        super().__init__()
        self._dim = dim
        self.a = uniform_rate / self.dim
        self.b = absorb_rate
        self.beta =  self.dim * self.a + self.b

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        edge_uniform = torch.ones(*i.shape, self.dim, device=i.device) 
        edge_uniform = edge_uniform.scatter(-1, i[..., None], - (self.dim - 1) )
        
        edge_absorb = F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)  
        return self.a * edge_uniform + self.b * edge_absorb
    
    def transp_rate(self, i):
        edge_uniform = torch.ones(*i.shape, self.dim, device=i.device)
        edge_uniform = edge_uniform.scatter(-1, i[..., None], - (self.dim - 1))

        edge_absorb = -F.one_hot(i, num_classes=self.dim)
        edge_absorb[i == self.dim - 1] += 1
        return self.a * edge_uniform + self.b * edge_absorb
    
    def transp_transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]*self.beta).exp()) * self.a / self.beta
        trans = trans.scatter(-1, i[..., None], torch.ones(*i.shape, self.dim, device=i.device) * ( self.a + ((self.dim-1)*self.a + self.b)*(-sigma[..., None]*self.beta).exp())/self.beta)
        trans += torch.where(
            i == self.dim - 1,
            (1 - (-sigma*self.beta).exp()) * self.b/self.beta,
            0
        )[..., None]
        return trans
    
    def transition(self, i, sigma):
        esigm1 = torch.where(
            sigma*self.beta < 0.5,
            torch.expm1(sigma*self.beta),
            torch.exp(sigma*self.beta) - 1
        )
        ratio = esigm1 / (1+esigm1)
        trans = torch.ones(*i.shape, self.dim, device=i.device) * ratio[..., None] * self.a / self.beta
        trans = trans.scatter(-1, i[..., None], torch.ones(*i.shape, self.dim, device=i.device) * ( self.a + ((self.dim-1)*self.a + self.b)*(1-ratio[..., None]))/self.beta)
        trans[..., -1] += ratio * self.b / self.beta
        return trans

    def sample_transition(self, i, sigma):
        absorb_prob = (1 - (-sigma*self.beta).exp()) * self.b / self.beta
        uniform_prob = (1 - (-sigma*self.beta).exp()) * self.a * self.dim / self.beta
        move_indices = torch.rand(*i.shape, device=i.device)
        i_pert = torch.where(move_indices < absorb_prob, self.dim - 1, i)
        i_pert = torch.where(move_indices > 1 - uniform_prob, torch.randint_like(i, self.dim), i_pert)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        epow = (dsigma*self.beta).exp()[..., None]
        extra_const = (1 - epow[...,0]) * self.b / self.beta * score.sum(dim=-1)
        
        score = ((1-epow)*self.a / self.beta) * score.sum(dim=-1, keepdim=True) + score * epow
        score[..., -1] += extra_const
        return score


    def sample_limit(self, *batch_dims):
        uniform_prob = self.a / self.beta
        return torch.where(
            torch.rand(*batch_dims) < uniform_prob,
            torch.randint(0, self.dim - 1, batch_dims, dtype=torch.int64),
            (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)
        )

    def score_entropy(self, score, sigma, x, x0):
        # esigm1 = torch.where(
        #     sigma < 0.5,
        #     torch.expm1(sigma),
        #     torch.exp(sigma) - 1
        # )
        # ratio = 1 - self.dim / (esigm1 + self.dim)

        # x: [B, L], x0: [B, L], score: [B, L, dim]
        weights = self.transition(x0, sigma) # [B, L, dim]
        # get prob indexed by x
        p0t = weights.gather(-1, x[..., None]).squeeze(-1) # [B, L]
        weights = weights.scatter(-1, x[..., None], torch.zeros_like(weights)) # [B, L, dim]
        weights = weights/p0t[..., None] # [B, L, dim]


        # negative term
        neg_term = (weights * score).mean(dim=-1)

        # positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim

        # constant factor
        const = weights.mean(dim=-1) 
        # print(pos_term.mean(), neg_term.mean(), const.mean())
        return pos_term - neg_term