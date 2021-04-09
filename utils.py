import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma


def sample_q(args, R, exact=True):

    if args.var_mode == 'nf_gamma':
        if exact:
            m = Gamma(args.lmbdas, args.betas)
            vs = m.sample(torch.Size([R])).squeeze(dim=2)
            xis = vs ** (1 / (2 * args.ks.repeat(1, R).T))
        else:
            shape = args.lmbdas.repeat(1, R).T
            rate = args.betas.repeat(1, R).T
            vs = gamma_icdf(shape=shape, rate=rate, args=args)
            k = args.ks.repeat(1, R).T
            xis = vs ** (1 / (2 * k)) # xis R by args.w_dim

    elif args.var_mode == 'nf_gammatrunc':

        m = Gamma(args.lmbdas, args.betas)
        vs = m.sample(torch.Size([R])).squeeze(dim=2)
        xis = vs ** (1 / (2 * args.ks.repeat(1, R).T))
        # TODO: need to check is in [0,args.xi_upper]

    elif args.var_mode == 'nf_gaussian':
        xis = torch.FloatTensor(R, args.w_dim).normal_(mean=0, std=1)

    return xis



def q_entropy_sample(args, xis):
    """

    :param args:
    :param xis: R by w_dim
    :return: estimate of E_q log q
    """

    if args.var_mode == 'nf_gaussian':

        q_rv = Normal(0, 1)
        return q_rv.log_prob(xis).sum(dim=1).mean()

    elif args.var_mode == 'nf_gamma' or args.var_mode == 'nf_gammatrunc':

        R = xis.shape[0]
        hs = args.hs.repeat(1,R).T
        betas = args.betas.repeat(1,R).T
        ks = args.ks.repeat(1,R).T
        return (hs*torch.log(xis)-betas*(xis**(2*ks))).mean(dim=0).sum() - qj_gengamma_lognorm(args.hs, args.ks, args.betas, args).sum()



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
def qj_gengamma_lognorm(h, k, beta, args):

    lmbda = (h + 1) / (2 * k)
    G = torch.lgamma(lmbda)

    if args.var_mode == 'nf_gamma':
        return G - torch.log(2*k) - lmbda*torch.log(beta)
    elif args.var_mode == 'nf_gammatrunc':
        return G - torch.log(2*k) - lmbda*torch.log(beta) + torch.log(torch.exp(G)-torch.igammac(lmbda,beta*(args.xi_upper**(2*k))))

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

