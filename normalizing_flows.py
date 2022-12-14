import torch
import torch.nn as nn
import numpy as np
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal


def setup_affinecoupling(nf_couplingpair, nf_hidden, w_dim):

    nets = lambda: nn.Sequential(nn.Linear(w_dim, nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(nf_hidden, nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(nf_hidden, nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(nf_hidden, w_dim), nn.Tanh())

    nett = lambda: nn.Sequential(nn.Linear(w_dim, nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(nf_hidden, nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(nf_hidden, nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(nf_hidden, w_dim))

    for layer in range(nf_couplingpair):
        ones = np.ones(w_dim)
        ones[np.random.choice(w_dim, w_dim // 2)] = 0
        half_mask = torch.cat((torch.from_numpy(ones.astype(np.float32)).unsqueeze(dim=0),
                               torch.from_numpy((1 - ones).astype(np.float32)).unsqueeze(dim=0)))
        if layer == 0:
            masks = half_mask
        else:
            masks = torch.cat((masks, half_mask))

    masks = nn.Parameter(masks, requires_grad=False)
    s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
    t = torch.nn.ModuleList([nett() for _ in range(len(masks))])

    return s, t, masks


# https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
#TODO: it woudl be nice to implement simpler NFs such as planar flows
class RealNVP(nn.Module):
    def __init__(self, base_dist, nf_couplingpair, nf_hidden, w_dim, sample_size, device=None, grad_flag='all'):
        super(RealNVP, self).__init__()

        # initialization of lmbda, k, beta in generalized gamma base distribution
        lmbda1 = torch.ones(1, 1)
        lmbdas_rest = torch.ones(w_dim - 1, 1)
        lmbdas = torch.cat((lmbda1, lmbdas_rest))

        k1 = torch.ones(1, 1)
        ks_rest = torch.ones(w_dim - 1, 1)
        ks = torch.cat((k1, ks_rest))

        beta1 = torch.ones(1, 1) * sample_size
        betas_rest = torch.ones(w_dim - 1, 1)*w_dim/2
        betas = torch.cat((beta1, betas_rest))

        if base_dist == 'gengamma':
            if grad_flag == 'all':

                self.lmbdas = torch.nn.Parameter(lmbdas, requires_grad=True)
                self.ks = torch.nn.Parameter(ks, requires_grad=True)
                self.betas = torch.nn.Parameter(betas, requires_grad=True)

            elif grad_flag == 'first':
                # TODO: not updating

                self.lmbda1 = torch.nn.Parameter(lmbda1, requires_grad=True)
                self.k1 = torch.nn.Parameter(k1, requires_grad=True)
                self.beta1 = torch.nn.Parameter(beta1, requires_grad=True)

                self.lmbdas = torch.cat((self.lmbda1, lmbdas_rest))
                self.ks = torch.cat((self.k1, ks_rest))
                self.betas = torch.cat((self.beta1, betas_rest))

            elif grad_flag == 'none':

                self.lmbdas = torch.nn.Parameter(lmbdas, requires_grad=False)
                self.ks = torch.nn.Parameter(ks, requires_grad=False)
                self.betas = torch.nn.Parameter(betas, requires_grad=False)

        if base_dist == 'gaussian_match':

            gengamma_d = 2*ks*lmbdas
            gengamma_a = betas**(-1/(2*ks))
            gengamma_p = 2*betas

            gengamma_mean = gengamma_a*torch.exp(torch.lgamma((gengamma_d+1)/gengamma_p) - torch.lgamma(gengamma_d/gengamma_p))

            term1 = torch.exp(torch.lgamma((gengamma_d+2)/gengamma_p) - torch.lgamma(gengamma_d/gengamma_p))
            gengamma_var = (gengamma_a**2)*(term1 - gengamma_mean**2)
            gengamma_logsigma = torch.log(gengamma_var**(1/2))

            if grad_flag == 'all':
                self.mu = torch.nn.Parameter(gengamma_mean.squeeze(dim=1), requires_grad=True)
                self.log_sigma = torch.nn.Parameter(gengamma_logsigma.squeeze(dim=1), requires_grad=True)
            elif grad_flag == 'first':

                # TODO: not updating
                self.mu1 = torch.nn.Parameter(gengamma_mean[0], requires_grad=True)
                self.log_sigma1 = torch.nn.Parameter(gengamma_logsigma[0], requires_grad=True)

                self.mu = torch.cat((self.mu1, gengamma_mean[1:, 0]))
                self.log_sigma = torch.cat((self.log_sigma1, gengamma_logsigma[1:, 0]))

            elif grad_flag == 'none':
                self.mu = torch.nn.Parameter(gengamma_mean.squeeze(dim=1), requires_grad=False)
                self.log_sigma = torch.nn.Parameter(gengamma_logsigma.squeeze(dim=1), requires_grad=False)

        elif base_dist == 'gaussian_std':

            if grad_flag == 'all':

                self.mu = torch.nn.Parameter(torch.zeros(w_dim), requires_grad=True)
                self.log_sigma = torch.nn.Parameter(torch.zeros(w_dim), requires_grad=True)

            elif grad_flag == 'first':

                # TODO: not updating
                self.mu1 = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
                self.log_sigma1 = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

                self.mu = torch.cat((self.mu1, torch.zeros(w_dim-1)))
                self.log_sigma = torch.cat((self.log_sigma1, torch.zeros(w_dim-1)))

            elif grad_flag == 'none':

                self.mu = torch.nn.Parameter(torch.zeros(w_dim), requires_grad=False)
                self.log_sigma = torch.nn.Parameter(torch.zeros(w_dim), requires_grad=False)

        self.s, self.t, self.masks = setup_affinecoupling(nf_couplingpair, nf_hidden, w_dim)

        self.w_dim = w_dim
        self.sample_size = sample_size
        self.device = device

    def forward(self, xi):

        log_det_J = xi.new_zeros(xi.shape[0])

        w = xi
        for i in range(len(self.t)):

            w_ = self.masks[i] * w
            s = self.s[i](w_)*(1 - self.masks[i])
            t = self.t[i](w_)*(1 - self.masks[i])
            w = (1 - self.masks[i]) * (w * torch.exp(s) + t) + w_
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

    def sample_xis(self, R, base_dist, upper=None):

        if base_dist == 'gengamma' or base_dist == 'gengammatrunc':

            betas = torch.abs(self.betas)
            ks = torch.abs(self.ks.repeat(1, R)).T
            lmbdas = torch.abs(self.lmbdas)

            m = Gamma(lmbdas, betas) # shape, rate
            vs = m.rsample(torch.Size([R])).squeeze(dim=2)
            xis = vs ** (1 / (2 * ks))

            if base_dist == 'gengammatrunc':
                xis = xis[torch.all(xis <= upper, dim=1), :]
                if xis.shape[0] == 0:
                    print('no xis')

        elif base_dist == 'gaussian_match' or base_dist == 'gaussian_std':

            xis = MultivariateNormal(self.mu, torch.diag(torch.exp(self.log_sigma)**2)).rsample(torch.Size([R]))

        xis = xis.to(self.device)

        return xis