import torch
import numpy as np
from torch.distributions.normal import Normal


def q_entropy_sample(args, xis):
    if args.var_mode == 'nf_gaussian':
        q_rv = Normal(0, 1)
        return q_rv.log_prob(xis).sum(dim=1).mean()

# q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'})
# E_{q_j} \log q_j = \frac{h_j'}{2k_j'} ( \psi(\lambda_j') - \log \beta_j ) - \lambda_j' - \log Z_j
def qj_entropy(args, lmbdas, ks, betas):

    if args.var_mode == 'nf_gamma':
        # logz = qj_gengamma_lognorm(hs, ks, betas)
        # return hs*(torch.digamma(lmbda) - torch.log(betas))/(2*ks) - lmbda - logz
        return -torch.lgamma(lmbdas) +torch.log(betas)/2*ks +torch.log(2*ks) - lmbdas + (lmbdas - 2*ks)*torch.digamma(lmbdas)
    elif args.var_mode == 'nf_gaussian':

        stds = 1 # TODO: should allow custom mean/std for nf_gaussian
        return -args.w_dim / 2 * np.log(2 * np.pi * np.e * (stds ** 2))


# normalizing constnat of q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'}): Z_j = \frac{\Gamma(\lambda_j')}{2k_j' \beta_j^{\lambda_j'}}
def qj_gengamma_lognorm(h, k, beta):
    lmbda = (h+1)/(2*k)
    G = torch.lgamma(lmbda)
    return G - torch.log(2*k) - lmbda*torch.log(beta)


# generate gamma(shape,rate)
def gamma_icdf(shape, rate, args):

    R = rate.shape[0]
    # TODO: add warning if not all entries in shape are the same
    if shape[0,0] < 36.0: #u is unif 0,1
        # inverse cdf of gamma with shape and rate
        # using approximation in Knowles
        u = torch.FloatTensor(R, args.w_dim).uniform_(0)
        g = torch.exp(torch.lgamma(shape))
        num = (u*shape*g)**(1/shape)
        return num/rate
    else: # here u is N(0,1)
        z = torch.FloatTensor(R, args.w_dim).normal_(mean=0,std=1)
        if ((shape+torch.sqrt(shape)*z)<0).sum() >0:
            print('warning xi generated negative')
        return (shape+torch.sqrt(shape)*z)/rate

