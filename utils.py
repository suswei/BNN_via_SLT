import torch
import numpy as np
from torch.distributions.normal import Normal

def q_entropy_sample(args, xis):

    if args.var_mode == 'nf_gaussian':

        q_rv = Normal(0, 1)
        return q_rv.log_prob(xis).sum(dim=1).mean()

    elif args.var_mode == 'nf_gamma':

        R = xis.shape[0]
        hs = args.hs.repeat(1,R).T
        betas = args.betas.repeat(1,R).T
        ks = args.ks.repeat(1,R).T
        return (hs*torch.log(xis)-betas*(xis**(2*ks))).mean(dim=0).sum() - qj_gengamma_lognorm(args.hs, args.ks, args.betas).sum()

    elif args.var_mode == 'nf_gammatrunc':

        ent = 0.0
        for dim in range(args.w_dim):
            ent += uni_trunc_gamma_entropy(args.betas[dim], args.lmbdas[dim], 1, xis[:, dim]) #TODO: allow custom upper limit
        return ent


# calculate entropy of univariate truncated gamma given sample of xi
# q(\xi) \propto \xi^{shape-1} exp(-rate \xi) 1(\xi \in [0,upper])
def uni_trunc_gamma_entropy(rate, shape, upper, xis):

    if torch.isinf(shape):
        entropy = -torch.log(upper)
    else:
        entropy = torch.log(trunc_gamma_pdf(xis, rate, shape, upper)).mean()
    return entropy


# q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'})
# E_{q_j} \log q_j = \frac{h_j'}{2k_j'} ( \psi(\lambda_j') - \log \beta_j ) - \lambda_j' - \log Z_j
def qj_entropy(args):

    if args.var_mode == 'nf_gamma':
        hs = args.hs
        ks = args.ks
        betas = args.betas
        lmbdas = (hs+1)/(2*ks)
        # logz = qj_gengamma_lognorm(hs, ks, betas)
        # return hs*(torch.digamma(lmbda) - torch.log(betas))/(2*ks) - lmbda - logz
        return -torch.lgamma(lmbdas) +torch.log(betas)/2*ks +torch.log(2*ks) - lmbdas + (lmbdas - 1/(2*ks))*torch.digamma(lmbdas)
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


# inverse cdf of gamma truncated to [0,b]
def trunc_gamma_icdf(u, upper, shape, rate):
    return gamma_icdf(u*torch.igamma(shape, rate*upper), shape, rate)


# density of gamma truncated to [0,b]
# http://www.m-hikari.com/astp/astp2013/astp21-24-2013/zaninettiASTP21-24-2013.pdf
def trunc_gamma_pdf(xi, rate, shape, upper):
    scale = 1/rate
    num = ((xi/scale)**(shape-1))*torch.exp(-xi/scale)
    G=torch.exp(torch.lgamma(1+shape))
    denom_k = scale*G*torch.igammac(1+shape,torch.Tensor([0.0]))- scale*G*torch.igammac(1+shape,upper/scale) + torch.exp(-upper/scale)*(scale**(-shape+1))*(upper**shape)
    return num*shape/denom_k