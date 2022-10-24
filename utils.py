import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
import torch.distributions as D


# TODO: implement mixture prior, log uniform prior, horseshoe prior
def log_prior(args, thetas):
    """

    :param args:
    :param thetas: [R, args.w_dim]
    :return: returns log varphi(theta_1), ..., log varphi(theta_R)
    """

    if args.prior == 'gaussian':

        return - args.w_dim/2*np.log(2*np.pi) \
               - (1/2)*args.w_dim*np.log(args.prior_var) \
               - torch.diag(torch.matmul(thetas-args.prior_mean, (thetas-args.prior_mean).T))/(2*args.prior_var)

    elif args.prior == 'logunif':

        a,b = 0.1,5
        prob = (thetas*np.log(b/a))**(-1)
        return torch.log(prob)

    elif args.prior == 'gmm':

        # mix = D.Categorical(torch.ones(2, ))
        mix = D.Categorical(torch.Tensor([0.5, 0.5]))
        comp = D.Independent(D.Normal(torch.zeros(2, args.w_dim), torch.cat((1e-2*torch.ones(1,args.w_dim),torch.ones(1,args.w_dim)),0)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm.log_prob(thetas)

    elif args.prior == 'unif':

        # return torch.log(1/(args.theta_upper-args.theta_lower)).sum() # assuming [-2,2]^d prior
        return torch.zeros(1)


def sample_q(resolution_network, args, R):

    if args.base_dist == 'gengamma' or args.base_dist == 'gengammatrunc':

        betas = torch.cat((args.sample_size * torch.ones(1, 1).to(args.device), resolution_network.betas))
        betas = torch.abs(betas)
        ks = torch.abs(resolution_network.ks.repeat(1, R)).T

        shape = torch.abs(resolution_network.lmbdas)
        m = Gamma(shape, betas)
        vs = m.rsample(torch.Size([R])).squeeze(dim=2)
        xis = vs ** (1 / (2 * ks))

        if args.base_dist == 'gengammatrunc':
            xis = xis[torch.all(xis <= args.upper, dim=1), :]
            if xis.shape[0] == 0:
                print('no xis')

    elif args.base_dist == 'gaussian':

        xis = torch.normal(args.nf_gaussian_mean, np.sqrt(args.nf_gaussian_var), size=(R, args.w_dim)).to(args.device)

    return xis


# def q_entropy_sample(args, xis, betas):
#     """
#
#     :param args:
#     :param xis: R by w_dim
#     :return: estimate of (scalar) E_q log q: 1/R \sum_{i=1}^R log q(xi_i)
#     """
#
#     if args.base_dist == 'gaussian':
#
#         q_rv = Normal(0, 1)
#         return q_rv.log_prob(xis).sum(dim=1).mean()
#
#     elif args.base_dist == 'gengamma' or args.base_dist == 'gengammatrunc':
#
#         R = xis.shape[0]
#         hs = args.hs.repeat(1,R).T
#         betas = betas.repeat(1,R).T
#         ks = args.ks.repeat(1,R).T
#         return (hs*torch.log(xis)-betas*(xis**(2*ks))).mean(dim=0).sum() - qj_gengamma_lognorm(args.hs, args.ks, args.betas, args).sum()


# q_j(\xi_j) \propto \xi_j^{h_j} \exp(-\beta_j \xi_j^{2k_j})
# E_{q_j} \log q_j = \frac{h_j}{2k_j} ( \psi(\lambda_j) - \log \beta_j ) - \lambda_j - \log Z_j
def Eqj_logqj(resolution_network, args):

    if args.base_dist == 'gengamma' or args.base_dist == 'gengammatrunc':

        betas = torch.cat((args.sample_size * torch.ones(1, 1).to(args.device), resolution_network.betas))
        betas = torch.abs(betas)
        ks = torch.abs(resolution_network.ks)
        lmbdas = torch.abs(resolution_network.lmbdas)

        if args.base_dist=='gengammatrunc':
            logZ = qj_gengamma_lognorm(lmbdas, ks, betas, trunc=True, b=args.upper)
        else:
            logZ = qj_gengamma_lognorm(lmbdas, ks, betas, trunc=False, b=None)

        return (lmbdas - 1 / (2 * ks))*(torch.digamma(lmbdas) - torch.log(betas)) - lmbdas - logZ

    elif args.base_dist == 'gaussian':

        return -args.w_dim / 2 * np.log(2 * np.pi * np.e * args.nf_gaussian_var)


# normalizing constnat of q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'}) supported on [0,b] where b could be infty
def qj_gengamma_lognorm(lmbdas, ks, betas, trunc=False, b=None):

    logZ = torch.lgamma(lmbdas) - torch.log(2*ks) - lmbdas*torch.log(betas)
    # if trunc:
    #     # TODO: torch.igamma: The backward pass with respect to first argument is not yet supported.
    #     return logZ + torch.log(torch.igamma(lmbdas, betas * (b ** (2 * ks))))
    # else:
    #     return logZ

    return logZ



