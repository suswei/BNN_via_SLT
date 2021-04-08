import os
import argparse
from torch.distributions.gamma import Gamma
import numpy as np
from scipy import stats
import torch.distributions as D

import custom_lr_scheduler
from dataset_factory import *
from gaussian_mf import *
from normalizing_flows import *
from utils import *
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt


# TODO: implement mixture prior, log uniform prior, horseshoe prior
# evaluate log varphi(theta), returns vector
def log_prior(args, thetas):

    if args.prior == 'gaussian':
    # varphi multivariate (dim=args.w_dim) Gaussian mean zero, covariance = diag(args.prior_var)
        return - args.w_dim/2*torch.log(2*torch.Tensor([np.pi])) \
               - (1/2)*args.w_dim*torch.log(torch.Tensor([args.prior_var])) \
               - torch.diag(torch.matmul(thetas,thetas.T))/(2*args.prior_var)
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
        return args.w_dim*np.log(1/4) # assuming [-2,2]^d prior


# TODO: implement student t
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
    elif args.var_mode == 'nf_gaussian':
        xis = torch.FloatTensor(R, args.w_dim).normal_(mean=0, std=1)

    return xis


def train(args):

    # resolution_network = R_NVP(d=args.w_dim, k=args.w_dim//2, hidden=args.nf_hidden, layers=args.nf_layers)
    resolution_network = RealNVP(dim=args.w_dim, hidden_dim=args.nf_hidden, layers=args.nf_layers, af=args.nf_af)
    optimizer = torch.optim.Adam(resolution_network.parameters(), lr=args.lr)
    scheduler = custom_lr_scheduler.CustomReduceLROnPlateau\
        (optimizer, 'min', verbose=True, factor=0.9, patience=100, eps=1e-6)

    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            pi = 2**(args.sample_size/args.batch_size-batch_idx)/(2**(args.sample_size/args.batch_size) - 1)
            resolution_network.train()
            optimizer.zero_grad()

            xis = sample_q(args, R=1, exact=True)

            # log_jacobians.mean() = E_q log |g'(xi)|
            thetas, log_jacobians = resolution_network(xis)

            # E_q \sum_i=1^m p(y_i |x_i , g(\xi))
            loglik_elbo_vec = loglik(thetas, data, target, args)

            # KL(q(\xi) || \varphi(g(\xi)) = E_q \log q - E_q log \varphi(g(\xi))) |g'(\xi)|
            complexity = q_entropy_sample(args, xis) - log_prior(args, thetas).mean() - log_jacobians.mean()

            elbo = loglik_elbo_vec.sum() - complexity/args.sample_size*args.batch_size
            running_loss += -elbo.item()

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:
            elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(resolution_network, args, R=1)
            elbo = elbo_loglik.mean() - complexity
            print('epoch {}: loss {}, nSn {}, elbo {} = loglik {} (loglik_val {}) - [complexity {} = qentropy {} - logprior {} - logjacob {}], '
                  .format(epoch, loss, args.nSn, elbo, elbo_loglik.mean(), elbo_loglik_val.mean(), complexity, ent, logprior.mean(), log_jacobians.mean()))

        scheduler.step(running_loss)
        #
        # if scheduler.has_convergence_been_reached():
        #     print('INFO: Converence has been reached. Stopping iterations.')
        #     break

    return resolution_network


def evaluate(resolution_network, args, R):

    resolution_network.eval()

    with torch.no_grad():

        xis = sample_q(args, R, exact=True)
        thetas, log_jacobians = resolution_network(xis)
        print('thetas min {} max {}'.format(thetas.min(), thetas.max()))
        # assuming xis \in [0,1]^d
        # theta1 = xis[:,0]
        # theta2 = 1.62167 / ((np.sqrt(2) * xis[:,1]) ** (-1) - 0.405963)
        #
        # plt.plot(xis[:, 1], xis[:, 2],'.')
        # plt.plot(thetas[:, 1], thetas[:, 2],'r.')
        # # plt.plot(theta1,theta2,'.')
        # plt.show()
        logprior = log_prior(args, thetas)
        ent = q_entropy_sample(args, xis)
        complexity = ent - logprior.mean() - log_jacobians.mean()

        # q_entropy_sample(args, xis)

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            elbo_loglik += loglik(thetas, data, target, args).sum(dim=1)

        elbo = elbo_loglik.mean() - complexity

        elbo_loglik_val = 0.0
        for batch_idx, (data, target) in enumerate(args.val_loader):
            elbo_loglik_val += loglik(thetas, data, target, args).sum(dim=1)

    return elbo_loglik.mean(), complexity, ent, logprior.mean(), log_jacobians.mean(), elbo_loglik_val.mean()


# for given sample size and supposed lambda, learn resolution map g and return acheived ELBO (plus entropy)
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--dataset', type=str, default='tanh',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['reducedrank', 'tanh'])

    parser.add_argument('--H', type=int, default=2)

    parser.add_argument('--sample_size', type=int, default=5000,
                        help='sample size of synthetic dataset')

    parser.add_argument('--prior', type=str)

    parser.add_argument('--prior_var', type=float, default=1e-3, metavar='N')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='N')

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 200)')

    parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 100)')

    parser.add_argument('--nf_hidden', type=int, default=16)

    parser.add_argument('--nf_layers', type=int, default=20)

    parser.add_argument('--lmbda_star', type=float, default=40, metavar='N',
                        help='?')

    parser.add_argument('--k', type=float, default=1, metavar='N',
                        help='?')

    parser.add_argument('--path', type=str)

    parser.add_argument('--var_mode', type=str, default='nf_gamma', choices=['nf_gamma','nf_gaussian','mf_gaussian'])

    parser.add_argument('--display_interval',type=int,default=500)

    parser.add_argument('--nf_af', type=str, default='relu',choices=['relu','tanh'])

    parser.add_argument('--beta_mode', type=str, default='lmbda_star', choices=['lmbda_star','ones'])

    args = parser.parse_args()

    get_dataset_by_id(args)

    print(args.path)
    print('true rlct {}'.format(args.trueRLCT))

    if args.var_mode == 'nf_gamma' or args.var_mode == 'nf_gaussian':

        # TODO: currently running nf_gamma with oracle lmbda value
        args.lmbda_star = get_lmbda([args.H], args.dataset)[0]
        args.lmbdas = args.lmbda_star*torch.ones(args.w_dim, 1)

        args.ks = args.k*torch.ones(args.w_dim, 1)
        args.hs = args.lmbdas*2*args.ks-1

        if args.beta_mode == 'lmbda_star':
            args.betas = args.lmbda_star*torch.ones(args.w_dim, 1)
        else:
            args.betas = torch.ones(args.w_dim, 1)
        args.betas[0] = args.sample_size

        print(args)

        net = train(args)
        elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=100)
        elbo = elbo_loglik.mean() - complexity

        print('exact elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
        print('-lambda log n + (m-1) log log n: {}'.format(-args.trueRLCT*np.log(args.sample_size) + (args.truem-1.0)*np.log(np.log(args.sample_size))))
        print('true lmbda {} versus supposed lmbda {}'.format(args.trueRLCT, args.lmbda_star))

        # i = 0
        # metric = []
        # for n in args.ns:
        #     args.betas[0] = n
        #     args.train_loader = args.datasets[i]
        #     elbo_loglik, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=10)
        #     complexity = qj_entropy(args).sum() - logprior - log_jacobians
        #     metric+= [elbo_loglik - complexity +args.nSns[i]]
        #     i+=1
        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(args.ns), metric)
        # print('est lmbda {} R2 {}'.format(-slope, r_value))

    elif args.var_mode == 'mf_gaussian':

        # TODO: might be out of date, especially w.r.t. prior
        print(args)
        net = train_pyvarinf(args)
        elbo, _, _ = evaluate_pyvarinf(net, args, R=10)

        print('exact elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
        print('-lambda log n + (m-1) log log n: {}'.format(
            -args.trueRLCT * np.log(args.sample_size) + (args.truem - 1.0) * np.log(np.log(args.sample_size))))
        print('true lmbda {}'.format(args.trueRLCT))

    results_dict = {'elbo': elbo,
                    'asy_log_pDn': -args.trueRLCT * np.log(args.sample_size) + (args.truem - 1.0) * np.log(
                        np.log(args.sample_size))}

    if args.path is not None:
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(vars(args), '{}/args.pt'.format(args.path))
        torch.save(net.state_dict(), '{}/state_dict.pt'.format(args.path))
        torch.save(results_dict, '{}/results.pt'.format(args.path))


if __name__ == "__main__":
    main()

