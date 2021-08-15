import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
import torch.distributions as D


# TODO: implement mixture prior, log uniform prior, horseshoe prior
# evaluate log varphi(theta), returns vector
def log_prior(args, thetas):
    """

    :param args:
    :param thetas: [R, args.w_dim]
    :return: returns log varphi(theta_1), ..., log varphi(theta_R)
    """

    if args.prior == 'gaussian':

        return - args.w_dim/2*np.log(2*np.pi) \
               - (1/2)*args.w_dim*np.log(args.prior_var) \
               - torch.diag(torch.matmul(thetas, thetas.T))/(2*args.prior_var)

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


def sample_q(resolution_network, args, R, exact=False):

    betas = torch.cat((args.sample_size * torch.ones(1, 1), resolution_network.betas))

    if args.method == 'nf_gamma':

        ks = torch.abs(resolution_network.ks.repeat(1, R)).T + 1

        if exact:
            m = Gamma(torch.abs(resolution_network.lmbdas)+1, betas)
            vs = m.sample(torch.Size([R])).squeeze(dim=2)
            xis = vs ** (1 / (2 * ks))
        else:
            shape = torch.abs(resolution_network.lmbdas.repeat(1, R)).T + 1
            rate = betas.repeat(1, R).T
            vs = gamma_icdf(shape=shape, rate=rate, args=args)
            xis = vs ** (1 / (2 * ks)) # xis R by args.w_dim

    # TODO: not currently reparametrizable, need to check is in [0, args.xi_upper]
    # elif args.method == 'nf_gammatrunc':
    #     m = Gamma(resolution_network.lmbdas, betas)
    #     vs = m.sample(torch.Size([R])).squeeze(dim=2)
    #     xis = vs ** (1 / (2 * resolution_network.ks.repeat(1, R).T))
    #

    elif args.method == 'nf_gaussian':
        # xis = torch.FloatTensor(R, args.w_dim).normal_(mean=0, std=1)
        xis = torch.FloatTensor(R, args.w_dim).normal_(mean=args.nf_gaussian_mean, std=np.sqrt(args.nf_gaussian_var))

        # m = Gamma(args.lmbdas[0], args.betas[0])
        # vs = m.sample(torch.Size([R]))
        # xis_beg = vs ** (1 / (2 * args.ks[0].repeat(1, R).T))
        # xis = torch.cat((xis_beg, xis_end),dim=1)

    return xis


# def q_entropy_sample(args, xis, betas):
#     """
#
#     :param args:
#     :param xis: R by w_dim
#     :return: estimate of (scalar) E_q log q: 1/R \sum_{i=1}^R log q(xi_i)
#     """
#
#     if args.method == 'nf_gaussian':
#
#         q_rv = Normal(0, 1)
#         return q_rv.log_prob(xis).sum(dim=1).mean()
#
#     elif args.method == 'nf_gamma' or args.method == 'nf_gammatrunc':
#
#         R = xis.shape[0]
#         hs = args.hs.repeat(1,R).T
#         betas = betas.repeat(1,R).T
#         ks = args.ks.repeat(1,R).T
#         return (hs*torch.log(xis)-betas*(xis**(2*ks))).mean(dim=0).sum() - qj_gengamma_lognorm(args.hs, args.ks, args.betas, args).sum()


# q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'})
# E_{q_j} \log q_j = \frac{h_j'}{2k_j'} ( \psi(\lambda_j') - \log \beta_j ) - \lambda_j' - \log Z_j
def exp_logqj(resolution_network, args):

    betas = torch.cat((args.sample_size * torch.ones(1, 1), resolution_network.betas))

    if args.method == 'nf_gamma':
        ks = torch.abs(resolution_network.ks)+1
        lmbdas = torch.abs(resolution_network.lmbdas)+1
        return -torch.lgamma(lmbdas) +torch.log(betas)/(2*ks) +torch.log(2*ks) \
               - lmbdas + (lmbdas - 1/(2*ks))*torch.digamma(lmbdas)

    elif args.method == 'nf_gaussian':

        return -args.w_dim / 2 * np.log(2 * np.pi * np.e * args.nf_gaussian_var)


# normalizing constnat of q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'}) supported on [0,b] where b could be infty
def qj_gengamma_lognorm(h, k, beta, args):

    lmbda = (h + 1) / (2 * k)
    G = torch.lgamma(lmbda)

    if args.method == 'nf_gamma':
        return G - torch.log(2*k) - lmbda*torch.log(beta)
    elif args.method == 'nf_gammatrunc':
        temp = beta * (args.xi_upper.unsqueeze(dim=1) ** (2 * k))
        return G - torch.log(2*k) - lmbda*torch.log(beta) + torch.log(torch.igamma(lmbda, temp))


# generate gamma(shape,rate) through inverse CDF

def gamma_icdf(shape, rate, args):

    R = rate.shape[0]

    small_shape_regime = shape < 5

    u = torch.FloatTensor(R, args.w_dim).uniform_(0)
    g = torch.exp(torch.lgamma(shape*small_shape_regime + (~small_shape_regime)))
    num = (u * shape * g) ** (1 / shape)
    small_shape = num/rate

    z = torch.FloatTensor(R, args.w_dim).normal_(mean=0, std=1)
    large_shape = (shape + torch.sqrt(shape) * z) / rate

    return small_shape_regime*small_shape + (~small_shape_regime)*large_shape

    # if shape[0,0] < 36.0: #u is unif 0,1
    #     # inverse cdf of gamma with shape and rate
    #     # using approximation in Knowles
    #     u = torch.FloatTensor(R, args.w_dim).uniform_(0)
    #     g = torch.exp(torch.lgamma(shape))
    #     num = (u*shape*g)**(1/shape)
    #     return num/rate
    # else: # here u is N(0,1)
    #     u = torch.FloatTensor(R, args.w_dim).normal_(mean=0, std=1)
    #     if ((shape + torch.sqrt(shape) * u) < 0).sum() > 0:
    #         print('warning xi generated negative')
    #     return (shape + torch.sqrt(shape) * u)/rate

