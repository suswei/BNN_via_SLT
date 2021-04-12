import os
import argparse
from dataset_factory import *
import custom_lr_scheduler
from gaussian_mf import *
from normalizing_flows import *
from utils import *
# from matplotlib import pyplot as plt
# from scipy import stats


def train(args):

    resolution_network = RealNVP(dim=args.w_dim, hidden_dim=args.nf_hidden, layers=args.nf_layers, af=args.nf_af)
    optimizer = torch.optim.Adam(resolution_network.parameters(), lr=args.lr)
    scheduler = custom_lr_scheduler.CustomReduceLROnPlateau\
        (optimizer, 'min', verbose=True, factor=0.9, patience=100, eps=1e-6)

    elbo_hist = []
    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            M = args.sample_size/args.batch_size # number of minibatches
            pi = (2**(M-batch_idx))/(2**M-1)
            resolution_network.train()
            optimizer.zero_grad()

            xis = sample_q(args, R=1, exact=True)

            thetas, log_jacobians = resolution_network(xis)  # log_jacobians.mean() sample estimate of E_q log |g'(xi)|
            args.theta_lower = thetas.min().detach()
            args.theta_upper = thetas.max().detach()

            loglik_elbo_vec = loglik(thetas, data, target, args)  # E_q \sum_i=1^m p(y_i |x_i , g(\xi))

            complexity = - log_prior(args, thetas).mean() - log_jacobians.mean()  # q_entropy no optimization

            # elbo = loglik_elbo_vec.sum() - complexity*(args.batch_size/args.sample_size)
            elbo = loglik_elbo_vec.sum() - complexity*pi

            running_loss += -elbo.item()

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:
            elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val \
                = evaluate(resolution_network, args, R=100)
            print('epoch {}: loss {}, nSn {}, elbo {} '
                  '= loglik {} (loglik_val {}) - [complexity {} = qentropy {} - logprior {} - logjacob {}], '
                  .format(epoch, loss, args.nSn, elbo, elbo_loglik.mean(), elbo_loglik_val.mean(), complexity, ent, logprior.mean(), log_jacobians.mean()))
            elbo_hist.append(elbo)

        scheduler.step(running_loss)

        if scheduler.has_convergence_been_reached():
            print('INFO: Converence has been reached. Stopping iterations.')
            break

    return resolution_network, elbo_hist


def evaluate(resolution_network, args, R):

    resolution_network.eval()

    with torch.no_grad():

        xis = sample_q(args, R, exact=True)
        thetas, log_jacobians = resolution_network(xis)

        print('thetas min {} max {}'.format(thetas.min(), thetas.max()))
        print('xis min {} max {}'.format(xis.min(), xis.max()))

        # assuming xis \in [0,1]^d
        # theta1 = xis[:,0]
        # theta2 = 1.62167 / ((np.sqrt(2) * xis[:,1]) ** (-1) - 0.405963)
        #
        # plt.plot(xis[:, 1], xis[:, 2],'.')
        # plt.plot(thetas[:, 1], thetas[:, 2],'r.')
        # plt.plot(theta1,theta2,'.')
        # plt.show()

        args.theta_lower = thetas.min()
        args.theta_upper = thetas.max()
        logprior = log_prior(args, thetas)

        args.xi_upper = xis.max()
        ent = q_entropy_sample(args, xis)

        complexity = ent - logprior.mean() - log_jacobians.mean()

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            elbo_loglik += loglik(thetas, data, target, args).sum(dim=1)

        elbo = elbo_loglik.mean() - complexity

        elbo_loglik_val = 0.0
        for batch_idx, (data, target) in enumerate(args.val_loader):
            elbo_loglik_val += loglik(thetas, data, target, args).sum(dim=1)

    return elbo, elbo_loglik.mean(), complexity, ent, logprior.mean(), log_jacobians.mean(), elbo_loglik_val.mean()


# for given sample size and supposed lambda, learn resolution map g and return acheived ELBO (plus entropy)
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--dataset', type=str, default='tanh',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['reducedrank', 'tanh'])

    parser.add_argument('--H', type=int, default=1)

    parser.add_argument('--sample_size', type=int, default=5000,
                        help='sample size of synthetic dataset')

    parser.add_argument('--prior', type=str, default='gaussian')

    parser.add_argument('--prior_var', type=float, default=1e-1, metavar='N')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='N')

    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 200)')

    parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 100)')

    parser.add_argument('--nf_hidden', type=int, default=16)

    parser.add_argument('--nf_layers', type=int, default=20)

    parser.add_argument('--nf_af', type=str, default='relu',choices=['relu','tanh'])

    parser.add_argument('--lmbda_star', type=float, default=40, metavar='N',
                        help='?')

    parser.add_argument('--k', type=float, default=1, metavar='N',
                        help='?')

    parser.add_argument('--path', type=str)

    parser.add_argument('--method', type=str, default='nf_gammatrunc', choices=['nf_gamma','nf_gammatrunc','nf_gaussian','mf_gaussian'])

    parser.add_argument('--display_interval',type=int, default=10)

    parser.add_argument('--varparams_mode', type=str, default='abs_gauss')

    args = parser.parse_args()

    get_dataset_by_id(args)

    print(args.path)
    print('true rlct {}'.format(args.trueRLCT))

    if args.method == 'nf_gamma' or args.method == 'nf_gaussian' or args.method == 'nf_gammatrunc':

        if args.varparams_mode == 'abs_gauss':
            args.lmbdas = 0.5*torch.ones(args.w_dim, 1)
            args.ks = torch.ones(args.w_dim, 1)
            args.betas = 0.5*torch.ones(args.w_dim, 1)
        elif args.varparams_mode == 'abs_gauss_n':
            lmbda_star = get_lmbda([args.H], args.dataset)[0]
            args.lmbdas = 0.5*torch.ones(args.w_dim, 1)
            args.lmbdas[0]=lmbda_star
            args.ks = torch.ones(args.w_dim, 1)
            args.betas = 0.5*torch.ones(args.w_dim, 1)
            args.betas[0] = args.sample_size
        elif args.varparams_mode == 'exp':
            args.lmbdas = torch.ones(args.w_dim, 1)
            args.ks = 0.5*torch.ones(args.w_dim, 1)
            args.betas = torch.ones(args.w_dim, 1)
        elif args.varparams_mode == 'icml':
            lmbda_star = get_lmbda([args.H], args.dataset)[0]
            # lmbda_star = 1000
            args.lmbdas = lmbda_star*torch.ones(args.w_dim, 1)
            args.ks = torch.ones(args.w_dim, 1)
            args.betas = lmbda_star*torch.ones(args.w_dim, 1)
            args.betas[0] = args.sample_size
        elif args.varparams_mode == 'allones':
            args.lmbdas = torch.ones(args.w_dim, 1)
            args.ks = torch.ones(args.w_dim, 1)
            args.betas = torch.ones(args.w_dim, 1)
        args.hs = args.lmbdas * 2 * args.ks - 1

        print(args)

        net, elbo_hist = train(args)
        elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=100)

        print('exact elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
        print('-lambda log n + (m-1) log log n: {}'.format(-args.trueRLCT*np.log(args.sample_size) + (args.truem-1.0)*np.log(np.log(args.sample_size))))
        print('true lmbda {} versus supposed lmbda {}'.format(args.trueRLCT, args.lmbda_star))

        # i = 0
        # metric = []
        # for n in args.ns:
        #     args.betas[0] = n
        #     args.train_loader = args.datasets[i]
        #     elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=10)
        #     metric+= [elbo_loglik - complexity +args.nSns[i]]
        #     i+=1
        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(args.ns), metric)
        # print('est lmbda {} R2 {}'.format(-slope, r_value))

    elif args.method == 'mf_gaussian':

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
                        np.log(args.sample_size)),
                    'elbo_hist': elbo_hist}

    if args.path is not None:
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(vars(args), '{}/args.pt'.format(args.path))
        torch.save(net.state_dict(), '{}/state_dict.pt'.format(args.path))
        torch.save(results_dict, '{}/results.pt'.format(args.path))


if __name__ == "__main__":
    main()

