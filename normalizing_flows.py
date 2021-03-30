import torch
import torch.nn
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

import itertools


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, layers):
        super().__init__()

        self.sizes = np.concatenate(
            ([in_dim], np.repeat(hidden_dim, layers + 1), [out_dim])).tolist()
        blocks = [[nn.Linear(in_f, out_f), nn.Tanh()]
                  for in_f, out_f in zip(self.sizes, self.sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last activation, don't need it in output layer

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)
        # for layer in self.layers:
        #     x=layer(x)


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.
    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim=8, layers=10, base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim, layers)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim, layers)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim, layers)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim, layers)

    def forward(self, x):
        lower, upper = x[:,:self.dim // 2], x[:,self.dim // 2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:,:self.dim // 2], z[:,self.dim // 2:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det




class R_NVP(nn.Module):
    def __init__(self, d, k, hidden, layers):
        super().__init__()
        self.d, self.k = d, k

        self.sizes = np.concatenate(
            ([k], np.repeat(hidden, layers + 1), [d-k])).tolist()
        blocks = [[nn.Linear(in_f, out_f), nn.LeakyReLU()]
                  for in_f, out_f in zip(self.sizes, self.sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last activation, don't need it in output layer

        self.sig_net = nn.Sequential(*blocks)

        self.mu_net = nn.Sequential(*blocks)

        # self.sig_net = nn.Sequential(
        #     nn.Linear(k, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, d - k))
        #
        # self.mu_net = nn.Sequential(
        #     nn.Linear(k, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, d - k))

        base_mu, base_cov = torch.zeros(d), torch.eye(d)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x, flip=True):
        x1, x2 = x[:, :self.k], x[:, self.k:]

        if flip:
            x2, x1 = x1, x2

        # forward
        sig = self.sig_net(x1)
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(x1)

        if flip:
            z2, z1 = z1, z2

        z_hat = torch.cat([z1, z2], dim=-1)

        # log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        # return z_hat, log_pz, log_jacob
        return z_hat, log_jacob

    def inverse(self, Z, flip=False):
        z1, z2 = Z[:, :self.k], Z[:, self.k:]

        if flip:
            z2, z1 = z1, z2

        x1 = z1
        x2 = (z2 - self.mu_net(z1)) * torch.exp(-self.sig_net(z1))

        if flip:
            x2, x1 = x1, x2
        return torch.cat([x1, x2], -1)