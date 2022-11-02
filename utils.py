import torch
import numpy as np
import torch.distributions as D


# TODO: implement mixture prior, log uniform prior, horseshoe prior
def log_prior(args, ws):
    """

    :param args:
    :param ws: [R, args.w_dim]
    :return: returns log varphi(theta_1), ..., log varphi(theta_R)
    """

    if args.prior == 'gaussian':

        return - args.w_dim/2*np.log(2*np.pi) \
               - (1/2)*args.w_dim*np.log(args.prior_var) \
               - torch.diag(torch.matmul(ws-args.prior_mean, (ws-args.prior_mean).T))/(2*args.prior_var)

    elif args.prior == 'logunif':

        a,b = 0.1,5
        prob = (ws*np.log(b/a))**(-1)
        return torch.log(prob)

    elif args.prior == 'gmm':

        # mix = D.Categorical(torch.ones(2, ))
        mix = D.Categorical(torch.Tensor([0.5, 0.5]))
        comp = D.Independent(D.Normal(torch.zeros(2, args.w_dim), torch.cat((1e-2*torch.ones(1,args.w_dim),torch.ones(1,args.w_dim)),0)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm.log_prob(ws)

    elif args.prior == 'unif':

        # return torch.log(1/(args.theta_upper-args.theta_lower)).sum() # assuming [-2,2]^d prior
        return torch.zeros(1)

# q_j(\xi_j) \propto \xi_j^{h_j} \exp(-\beta_j \xi_j^{2k_j})
# E_{q_j} \log q_j = \frac{h_j}{2k_j} ( \psi(\lambda_j) - \log \beta_j ) - \lambda_j - \log Z_j
def Eqj_logqj(resolution_network, args):

    if args.base_dist == 'gengamma' or args.base_dist == 'gengammatrunc':

        betas = torch.cat((args.sample_size * torch.ones(1, 1).to(args.device), resolution_network.betas))
        betas = torch.abs(betas)
        ks = torch.abs(resolution_network.ks)
        lmbdas = torch.abs(resolution_network.lmbdas)

        if args.base_dist == 'gengammatrunc':
            logZ = qj_gengamma_lognorm(lmbdas, ks, betas, b=args.upper)
        else:
            logZ = qj_gengamma_lognorm(lmbdas, ks, betas, b=None)

        return (lmbdas - 1 / (2 * ks))*(torch.digamma(lmbdas) - torch.log(betas)) - lmbdas - logZ

    elif args.base_dist == 'gaussian_match' or args.base_dist == 'gaussian_std':

        return -1 / 2 * torch.log(2 * np.pi * torch.exp(resolution_network.log_sigma)**2) - 1/2


# normalizing constnat of q_j(\xi_j) \propto \xi_j^{h_j'} \exp(-\beta_j \xi_j^{2k_j'}) supported on [0,b] where b could be infty
def qj_gengamma_lognorm(lmbdas, ks, betas, b=None):

    logZ = torch.lgamma(lmbdas) - torch.log(2*ks) - lmbdas*torch.log(betas)
    # TODO: torch.igamma: The backward pass with respect to first argument is not yet supported. should there be a warning in the meantime?
    # TODO: what if we free lmbdas and take advantage of igamma having backward pass wrt second argument?
    if b is not None:
        return logZ + torch.log(torch.igamma(lmbdas, betas * (b ** (2 * ks))))
    else:
        return logZ

    return logZ



