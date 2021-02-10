import os
import argparse
from torch.distributions.gamma import Gamma
import numpy as np

import custom_lr_scheduler
from dataset_factory import *
from gaussian_mf import *
from normalizing_flows import RealNVP
from utils import *


# varphi multivariate (dim=args.w_dim) Gaussian mean zero, covariance = diag(args.prior_var)
# evaluate log varphi(theta), returns vector
def varphi_logprob(args, thetas):
    return - args.w_dim/2*torch.log(2*torch.Tensor([np.pi])) \
           - (1/2)*args.w_dim*torch.log(torch.Tensor([args.prior_var])) \
           - torch.diag(torch.matmul(thetas,thetas.T))/(2*args.prior_var)


def train(args):

    resolution_network = RealNVP(dim=args.w_dim, hidden_dim=args.nf_hidden, layers=args.nf_layers)
    optimizer = torch.optim.Adam(resolution_network.parameters(), lr=args.lr)
    scheduler = custom_lr_scheduler.CustomReduceLROnPlateau\
        (optimizer, 'min', verbose=True, factor=0.9, patience=100, eps=1e-6)

    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            resolution_network.train()
            optimizer.zero_grad()

            # log_jacobians.mean() = E_q log |g'(xi)|
            shape = args.lmbdas.repeat(1, args.R).T
            rate = args.betas.repeat(1, args.R).T
            vs = gamma_icdf(shape=shape, rate=rate, args=args)
            k = args.ks.repeat(1, args.R).T
            xis = vs ** (1 / (2 * k))

            # xis R by args.w_dim
            thetas, log_jacobians = resolution_network(xis)  # g

            # E_q 1/m \sum_i=1^m p(y_i |x_i , g(\xi))
            loglik_elbo = loglik(thetas, data, target, args, R=args.R).mean()

            # KL(q(\xi) || \varphi(g(\xi)) = E_q \log q - E_q log \varphi(g(\xi)))
            # q(\xi_1,...,\xi_d) = q(\xi_1)*...*q(\xi_d)
            # E_q log q = \sum_j=1^d E_qj \log qj
            # E_q log q(xi)/varphi(g(xi))
            prior_elbo = args.qentropy - varphi_logprob(args, thetas).mean()
            elbo = loglik_elbo + (log_jacobians.mean() - prior_elbo)/args.sample_size
            running_loss += loglik_elbo * args.batch_size / args.sample_size + (log_jacobians.mean() - prior_elbo)/args.sample_size

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:
            elbo, elbo1, elbo2, elbo3 = evaluate(resolution_network, args, R=1)
            print('epoch {}: loss {}, nSn {}, exact elbo {} = loglik {} + jacob {} - prior {}'
                  .format(epoch, loss, args.nSn, elbo, elbo1, elbo2, elbo3))

        scheduler.step(running_loss.item())

        if scheduler.has_convergence_been_reached():
            print('INFO: Converence has been reached. Stopping iterations.')
            break

    return resolution_network


def evaluate(resolution_network, args, R):

    resolution_network.eval()

    with torch.no_grad():
        # generate xis R by args.w_dim
        args.betas[0] = args.sample_size
        m = Gamma(args.lmbdas, args.betas)
        vs = m.sample(torch.Size([R])).squeeze(dim=2)
        xis = vs ** (1 / (2 * args.ks.repeat(1, R).T))

        thetas, log_jacobians = resolution_network(xis)


        prior_elbo = qj_entropy(args.hs, args.ks, args.betas).sum() - varphi_logprob(args, thetas).mean()

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            elbo_loglik += loglik(thetas, data, target, args, R=R).sum(dim=1)

        elbo = elbo_loglik.mean() + log_jacobians.mean() - prior_elbo

    return elbo, elbo_loglik.mean(), log_jacobians.mean(), prior_elbo


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

    parser.add_argument('--lr', type=float, default=1e-3, metavar='N')

    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 200)')

    parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 100)')

    parser.add_argument('--nf_hidden', type=int, default=8)

    parser.add_argument('--nf_layers', type=int, default=10)

    parser.add_argument('--R', type=int, default=1, metavar='N',
                        help='?')

    parser.add_argument('--lmbda_star', type=float, default=1, metavar='N',
                        help='?')

    parser.add_argument('--k', type=float, default=1, metavar='N',
                        help='k associated with global RLCT')

    parser.add_argument('--path', type=str)

    parser.add_argument('--mf_mode', type=str, default='gengamma', choices=['gengamma','gaussian'])

    parser.add_argument('--display_interval',type=int,default=500)

    args = parser.parse_args()


    get_dataset_by_id(args)
    args.prior_var = 1/args.H

    print(args.path)
    print('true rlct {}'.format(args.trueRLCT))


    if args.mf_mode == 'gengamma':

        print(args)

        betas = args.lmbda_star*torch.ones(args.w_dim, 1)
        print('non beta* set to {}'.format(args.lmbda_star))
        betas[0] = args.sample_size


        lmbdas = args.lmbda_star*torch.ones(args.w_dim, 1) # other lambdas should be >= global lambda
        print('all lambdas set to conjectured lambda {}'.format(args.lmbda_star))

        ks = torch.ones(args.w_dim, 1)
        ks[0] = args.k
        hs = lmbdas*2*ks-1

        args.hs = hs
        args.ks = ks
        args.betas = betas
        args.lmbdas = lmbdas

        args.qentropy = qj_entropy(args.hs, args.ks, args.betas).sum()

        net = train(args)
        elbo, _, _, _ = evaluate(net, args, R=100)

        print('exact elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
        print('-lambda log n + (m-1) log log n: {}'.format(-args.trueRLCT*np.log(args.sample_size) + (args.truem-1.0)*np.log(np.log(args.sample_size))))
        print('true lmbda {} versus supposed lmbda {}'.format(args.trueRLCT, args.lmbda_star))

    elif args.mf_mode == 'gaussian':

        print(args)
        net = train_pyvarinf(args)
        elbo, _, _ = evaluate_pyvarinf(net, args, R=100)

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

