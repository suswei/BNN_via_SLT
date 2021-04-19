import torch
import torch.nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

import itertools


import torch.nn.functional as F
import math

import torch as t
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class AutoregressiveLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = Parameter(t.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(t.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return t.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ f(G(x)) + (1 - σ(x)) ⨀ Q(x) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class IAF(nn.Module):
    def __init__(self, latent_size, h_size=None):
        super(IAF, self).__init__()

        self.z_size = latent_size
        self.h_size = h_size

        self.h = Highway(self.h_size, 3, nn.ELU())

        self.m = nn.Sequential(
            AutoregressiveLinear(self.z_size + self.h_size, self.z_size),
            # AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size)
        )

        self.s = nn.Sequential(
            AutoregressiveLinear(self.z_size + self.h_size, self.z_size),
            # AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size)
        )

    def forward(self, z, h=None):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the IAF mapping Jacobian
        """

        if h is None:
            h = torch.zeros(z.shape[0],self.h_size)
        h = self.h(h)

        input = t.cat([z, h], 1)
        # input = z
        m = self.m(input)
        s = self.s(input)

        z = s.exp() * z + m

        log_det = s.sum(1)

        return z, log_det


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, layers, af):
        super().__init__()

        self.sizes = np.concatenate(
            ([in_dim], np.repeat(hidden_dim, layers + 1), [out_dim])).tolist()
        if af == 'relu':
            blocks = [[nn.Linear(in_f, out_f), nn.ReLU()] #TODO: does the activation matter?
                      for in_f, out_f in zip(self.sizes, self.sizes[1:])]
        elif af == 'tanh':
            blocks = [[nn.Linear(in_f, out_f), nn.Tanh()] #TODO: does the activation matter?
                      for in_f, out_f in zip(self.sizes, self.sizes[1:])]
        blocks = list(itertools.chain(*blocks))
        del blocks[-1]  # remove the last activation, don't need it in output layer

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)
        # for layer in self.layers:
        #     x=layer(x)


# TODO: jacobian quickly becomes constant for different xis when layer is large
class RealNVP(nn.Module):
    """
    Non-volume preserving flow.
    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim=8, layers=10, base_network=FCNN, af='relu'):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim, layers, af)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim, layers, af)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim, layers, af)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim, layers, af)

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

        blocks2 = [[nn.Linear(in_f, out_f), nn.LeakyReLU()]
                  for in_f, out_f in zip(self.sizes, self.sizes[1:])]
        blocks2 = list(itertools.chain(*blocks2))
        del blocks2[-1]  # remove the last activation, don't need it in output layer

        self.sig_net = nn.Sequential(*blocks)
        print(self.sig_net)
        self.mu_net = nn.Sequential(*blocks2)
        print(self.mu_net)

        # self.sig_net = nn.Sequential(
        #     nn.Linear(k, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, d - k))
        #
        # self.mu_net = nn.Sequential(
        #     nn.Linear(k, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, d - k))

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

    def inverse(self, Z, flip=True):
        z1, z2 = Z[:, :self.k], Z[:, self.k:]

        if flip:
            z2, z1 = z1, z2

        x1 = z1
        x2 = (z2 - self.mu_net(z1)) * torch.exp(-self.sig_net(z1))

        if flip:
            x2, x1 = x1, x2
        return torch.cat([x1, x2], -1)