import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import math
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


# https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, d, trueRLCT):
        super(RealNVP, self).__init__()

        # initialize at N(lambda/beta, lambda/beta**2)**(1/2k) = N(1, 1/trueRLCT)
        self.lmbdas = torch.nn.Parameter(torch.cat((torch.ones(1,1)*trueRLCT, torch.rand(d-1, 1)+trueRLCT)), requires_grad=True)
        # self.ks = torch.nn.Parameter(torch.cat((torch.rand(1, 1)+0.5, torch.rand(d-1, 1)+0.5)), requires_grad=True)
        self.ks = torch.nn.Parameter(torch.ones(d, 1)/2, requires_grad=True) #decreasing k will increase variance
        self.betas = torch.nn.Parameter(torch.rand(d-1, 1)+trueRLCT, requires_grad=True)

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

        self.d = d

        # hidden = 256
        # self.sig_net = nn.Sequential(
        #     nn.Linear(self.d-1, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, 1),
        #     nn.Sigmoid()
        # )
        #
        # self.mu_net = nn.Sequential(
        #     nn.Linear(self.d-1, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, 1),
        #     nn.Sigmoid()
        # )
    # def inverse(self, z):
    #     x = z
    #     for i in range(len(self.t)):
    #         x_ = x * self.mask[i]
    #         s = self.s[i](x_) * (1 - self.mask[i])
    #         t = self.t[i](x_) * (1 - self.mask[i])
    #         x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
    #     return x

    def forward(self, xi):

        log_det_J = xi.new_zeros(xi.shape[0])

        # xi1, xi2 = xi[:, :1], xi[:, 1:]
        # sig = self.sig_net(xi2)
        # u1, u2 = xi1 * torch.exp(sig) + self.mu_net(xi2), xi2
        # u = torch.cat([u1, u2], dim=1)
        # log_det_J += sig.sum(-1)

        w = xi
        for i in range(len(self.t)):

            w_ = self.mask[i] * w

            # s = self.s[i](w_) * (1 - self.mask[i])
            # t = self.t[i](w_) * (1 - self.mask[i])
            # w = (1 - self.mask[i]) * (w - t) * torch.exp(-s) + w_
            # log_det_J -= s.sum(dim=1)

            s = self.s[i](w_)*(1 - self.mask[i])
            t = self.t[i](w_)*(1 - self.mask[i])
            w = (1 - self.mask[i]) * (w * torch.exp(s) + t) + w_
            log_det_J += s.sum(dim=1)

        return w, log_det_J

    # def log_prob(self, x):
    #     z, logp = self.f(x)
    #     return self.prior.log_prob(z) + logp
    #
    # def sample(self, batchSize):
    #     z = self.prior.sample((batchSize, 1))
    #     logp = self.prior.log_prob(z)
    #     x = self.g(z)
    #     return x