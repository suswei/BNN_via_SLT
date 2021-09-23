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

    if args.method == 'nf_gamma' or args.method == 'nf_gammatrunc':

        betas = torch.cat((args.sample_size * torch.ones(1, 1).to(args.device), resolution_network.betas))
        betas = torch.abs(betas)
        ks = torch.abs(resolution_network.ks.repeat(1, R)).T

        if exact:
            shape = torch.abs(resolution_network.lmbdas)
            m = Gamma(shape, betas)
            vs = m.sample(torch.Size([R])).squeeze(dim=2)
            xis = vs ** (1 / (2 * ks))

        else:
            shape = torch.abs(resolution_network.lmbdas.repeat(1, R)).T
            rate = betas.repeat(1, R).T
            vs = gamma_icdf(shape=shape, rate=rate, args=args)
            r = torch.nn.ReLU()
            vs = r(vs)
            if torch.any(torch.isnan(vs)):
                print('nan xis')
            if torch.any(torch.isnan(ks)):
                print('nan ks')
            xis = vs ** (1 / (2 * ks)) # xis R by args.w_dim

        if args.method == 'nf_gammatrunc':
            xis = xis[torch.all(xis <= args.upper, dim=1),:]
            if len(xis.size()) == 0:
                print('no xis')

    elif args.method == 'nf_gaussian':
        # xis = torch.FloatTensor(R, args.w_dim).normal_(mean=0, std=1)
        xis = torch.normal(args.nf_gaussian_mean, np.sqrt(args.nf_gaussian_var), size=(R, args.w_dim)).to(args.device)
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


# q_j(\xi_j) \propto \xi_j^{h_j} \exp(-\beta_j \xi_j^{2k_j})
# E_{q_j} \log q_j = \frac{h_j}{2k_j} ( \psi(\lambda_j) - \log \beta_j ) - \lambda_j - \log Z_j
def Eqj_logqj(resolution_network, args):

    if args.method == 'nf_gamma' or args.method == 'nf_gammatrunc':

        betas = torch.cat((args.sample_size * torch.ones(1, 1).to(args.device), resolution_network.betas))
        betas = torch.abs(betas)
        ks = torch.abs(resolution_network.ks)
        lmbdas = torch.abs(resolution_network.lmbdas)

        if args.method=='nf_gammatrunc':
            logZ = qj_gengamma_lognorm(lmbdas, ks, betas, trunc=True, b=args.upper)

        else:
            logZ = qj_gengamma_lognorm(lmbdas, ks, betas, trunc=False, b=None)

        return (lmbdas - 1 / (2 * ks))*(torch.digamma(lmbdas) - torch.log(betas)) - lmbdas - logZ
        # return -torch.lgamma(lmbdas) + torch.log(betas)/(2*ks) + torch.log(2*ks) \
        #        - lmbdas + (lmbdas - 1/(2*ks))*torch.digamma(lmbdas)

    elif args.method == 'nf_gaussian':

        return -args.w_dim / 2 * np.log(2 * np.pi * np.e * args.nf_gaussian_var)


# normalizing constnat of q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'}) supported on [0,b] where b could be infty
def qj_gengamma_lognorm(lmbdas, ks, betas, trunc=False, b=None):

    logZ = torch.lgamma(lmbdas) - torch.log(2*ks) - lmbdas*torch.log(betas)

    if trunc:
        # torch.igamma: The backward pass with respect to first argument is not yet supported.
        return logZ + torch.log(torch.igamma(lmbdas, betas * (b ** (2 * ks))))
    else:
        return logZ


# generate gamma(shape,rate) through inverse CDF

def gamma_icdf(shape, rate, args):
# only implementing large shape case

    R = rate.shape[0]

    # small_shape_regime = shape < 4
    #
    # u = torch.rand(R, args.w_dim).to(args.device)
    # # u = torch.FloatTensor(R, args.w_dim).uniform_(0).to(args.device)
    # g = torch.exp(torch.lgamma(shape*small_shape_regime + (~small_shape_regime)))
    # num = (u * shape * g) ** (1 / shape)
    # small_shape = num/rate

    z = torch.normal(0, 1, size=(R, args.w_dim)).to(args.device)
    # z = torch.Tensor(R, args.w_dim).normal_(mean=0, std=1).to(args.device)
    large_shape = (shape + torch.sqrt(shape) * z) / rate

    # vs = small_shape_regime*small_shape + (~small_shape_regime)*large_shape
    vs = large_shape

    return vs



