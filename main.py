import os
import argparse
from torch.distributions.gamma import Gamma
import numpy as np
from scipy import stats

import custom_lr_scheduler
from dataset_factory import *
from gaussian_mf import *
from normalizing_flows import *
from utils import *
from torch.distributions.normal import Normal


# TODO: implement mixture prior
# evaluate log varphi(theta), returns vector
def log_prior(args, thetas):

    # if args.prior == 'gaussian':
    # varphi multivariate (dim=args.w_dim) Gaussian mean zero, covariance = diag(args.prior_var)
    return - args.w_dim/2*torch.log(2*torch.Tensor([np.pi])) \
           - (1/2)*args.w_dim*torch.log(torch.Tensor([args.prior_var])) \
           - torch.diag(torch.matmul(thetas,thetas.T))/(2*args.prior_var)
    # elif args.prior == 'uniform':
    #     return np.log(1/5)*torch.ones(args.w_dim,1)


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
        inv_temp = min(1, 0.01 + epoch / args.epochs)

        for batch_idx, (data, target) in enumerate(args.train_loader):

            pi = 2**(args.sample_size/args.batch_size-batch_idx)/(2**(args.sample_size/args.batch_size) - 1)
            resolution_network.train()
            optimizer.zero_grad()

            xis = sample_q(args, R=1, exact=True)

            # log_jacobians.mean() = E_q log |g'(xi)|
            thetas, log_jacobians = resolution_network(xis)

            # E_q 1/m \sum_i=1^m p(y_i |x_i , g(\xi))
            loglik_elbo_vec = loglik(thetas, data, target, args)

            # KL(q(\xi) || \varphi(g(\xi)) = E_q \log q - E_q log \varphi(g(\xi)))
            # complexity = args.qentropy - log_prior(args, thetas).mean() - log_jacobians.mean()
            complexity = - log_prior(args, thetas).mean() - log_jacobians.mean()

            # if args.var_mode == 'nf_gaussian':
            #     complexity = q_entropy_sample(args, xis) - log_prior(args, thetas).mean() - log_jacobians.mean()

            # elbo = loglik_elbo_vec.mean() - complexity/args.sample_size
            elbo = loglik_elbo_vec.sum() - pi*complexity # section 3.4 of https://arxiv.org/pdf/1505.05424.pdf

            # running_loss += -loglik_elbo_vec.mean() * args.batch_size / args.sample_size + complexity/args.sample_size

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:
            elbo_loglik, logprior, log_jacobians, elbo_loglik_val = evaluate(resolution_network, args, R=1)
            complexity = qj_entropy(args).sum() - logprior.mean() - log_jacobians.mean()
            elbo = elbo_loglik.mean() - complexity
            print('epoch {}: loss {}, nSn {}, elbo {} = loglik {} (loglik_val {}) - [complexity {} = qentropy {} - logprior {} - logjacob {}], '
                  .format(epoch, loss, args.nSn, elbo, elbo_loglik.mean(), elbo_loglik_val.mean(), complexity, qj_entropy(args).sum(), logprior.mean(), log_jacobians.mean()))

            scheduler.step(-elbo)

            if scheduler.has_convergence_been_reached():
                print('INFO: Converence has been reached. Stopping iterations.')
                break

    return resolution_network


def evaluate(resolution_network, args, R):

    resolution_network.eval()

    with torch.no_grad():

        xis = sample_q(args, R, exact=True)
        thetas, log_jacobians = resolution_network(xis)
        logprior = log_prior(args, thetas)
        complexity = qj_entropy(args).sum() - logprior.mean() - log_jacobians.mean()

        # q_entropy_sample(args, xis)

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            elbo_loglik += loglik(thetas, data, target, args).sum(dim=1)

        elbo = elbo_loglik.mean() - complexity

        elbo_loglik_val = 0.0
        for batch_idx, (data, target) in enumerate(args.val_loader):
            elbo_loglik_val += loglik(thetas, data, target, args).sum(dim=1)

    return elbo_loglik.mean(), logprior.mean(), log_jacobians.mean(), elbo_loglik_val.mean()


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

    args = parser.parse_args()

    get_dataset_by_id(args)
    args.prior_var = 1/args.H
    # args.prior_var = 1e-2

    print(args.path)
    print('true rlct {}'.format(args.trueRLCT))

    if args.var_mode == 'nf_gamma' or args.var_mode == 'nf_gaussian':


        # TODO: currently running nf_gamma with oracle lmbda value
        args.lmbda_star = get_lmbda([args.H], args.dataset)[0]
        args.lmbdas = torch.ones(args.w_dim, 1)
        # args.lmbdas[0] = args.lmbda_star

        args.ks = args.k*torch.ones(args.w_dim, 1)
        args.hs = args.lmbdas*2*args.ks-1

        # args.betas = torch.exp(2*args.ks* (-args.hs*torch.digamma(args.lmbdas)/(2*args.ks) + args.lmbdas + torch.lgamma(args.lmbdas) - torch.log(2*args.ks) - args.w_dim) ) # designed to make normalizing constant of q_j = 1
        args.betas = torch.ones(args.w_dim, 1)


        qj_entropy(args).sum()
        print(args)


        net = train(args)
        elbo_loglik, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=100)
        elbo = elbo_loglik.mean() - (qj_entropy(args).sum()-logprior.mean()-log_jacobians.mean())

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

