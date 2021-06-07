import os
import argparse
from dataset_factory import *
import custom_lr_scheduler
from normalizing_flows import *
from utils import *


def set_gengamma_varparams(args):

    # constrain lambda_j = beta_j for j \ne g
    # further enforce log beta_j - digamma(lambda_j) small for j \ne g
    # this has the effect of forcing lambda_j = beta_j = large value for j \ne g
    # let lambda_g be zero of digamma
    # enforce all k_j to be such that lambda*(digamma(lambda)-1) + log(2k) - lgamma(lambda) = 0
    a0 = 1.461632
    args.lmbdas = a0*torch.ones(args.w_dim, 1)
    # args.lmbdas = 0.5*torch.ones(args.w_dim,1)
    args.lmbdas[0] = args.lmbda0
    args.betas = a0*torch.ones(args.w_dim, 1)
    if args.betastar == 'true':
        args.betas[0] = args.sample_size
    args.ks = torch.exp(a0+torch.lgamma(a0*torch.ones(1)))*torch.ones(args.w_dim, 1)/2
    args.ks[0] = args.k0

    args.hs = args.lmbdas * 2 * args.ks - 1


def train(args):


    nets = lambda: nn.Sequential(nn.Linear(args.w_dim, args.nf_hidden), nn.LeakyReLU(),
                             nn.Linear(args.nf_hidden, args.nf_hidden), nn.LeakyReLU(),
                             nn.Linear(args.nf_hidden, args.nf_hidden), nn.LeakyReLU(),
                             nn.Linear(args.nf_hidden, args.w_dim), nn.Tanh())

    nett = lambda: nn.Sequential(nn.Linear(args.w_dim, args.nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(args.nf_hidden, args.nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(args.nf_hidden, args.nf_hidden), nn.LeakyReLU(),
                                 nn.Linear(args.nf_hidden, args.w_dim))

    for layer in range(args.no_couplingpairs):
        ones = np.ones(args.w_dim)
        ones[np.random.choice(args.w_dim, args.w_dim // 2)] = 0
        half_mask = torch.cat((torch.from_numpy(ones.astype(np.float32)).unsqueeze(dim=0),
                               torch.from_numpy((1 - ones).astype(np.float32)).unsqueeze(dim=0)))

        if layer == 0:
            masks = half_mask
        else:
            masks = torch.cat((masks, half_mask))

    resolution_network = RealNVP(nets, nett, masks, args.w_dim)
    resolution_network.to(args.device)

    optimizer = torch.optim.Adam(resolution_network.parameters(), lr=args.lr)
    scheduler = custom_lr_scheduler.CustomReduceLROnPlateau\
        (optimizer, 'min', verbose=True, factor=0.9, patience=100, eps=1e-6)

    elbo_hist = []
    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            data, target = data.to(args.device), target.to(args.device)

            resolution_network.train()
            optimizer.zero_grad()

            if args.prior == 'unif':
                args.trainR = 10

            xis = sample_q(args, args.trainR, exact=True)  # [R, args.w_dim]
            xis = xis.to(args.device)

            thetas, log_jacobians = resolution_network(xis)  # log_jacobians [R, 1]  E_q log |g'(xi)|

            args.theta_lower = torch.min(thetas, dim=0).values.detach()
            args.theta_upper = torch.max(thetas, dim=0).values.detach()
            # print(thetas.mean(dim=0))

            loglik_elbo_vec = loglik(thetas, data, target, args)  # [R, minibatch_size] E_q \sum_i=1^m p(y_i |x_i , g(\xi))

            complexity = - log_prior(args, thetas, xis).mean() - log_jacobians.mean()  # q_entropy no optimization

            if args.blundell_weighting:
                M = args.sample_size/args.batch_size # number of minibatches
                pi = (2**(M-batch_idx))/(2**M-1) # follows blundell, is bad for nf_gammatrunc
                elbo = loglik_elbo_vec.mean(dim=0).sum() - complexity*pi
            else:
                elbo = loglik_elbo_vec.mean(dim=0).sum() - complexity * (args.batch_size / args.sample_size)

            running_loss += -elbo.item()

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:
            evalR = 10
            elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val \
                = evaluate(resolution_network, args, R=evalR)
            print('epoch {}: loss {}, nSn {}, (R = {}) elbo {} '
                  '= loglik {} (loglik_val {}) - [complexity {} = qentropy {} - logprior {} - logjacob {} ], '
                  .format(epoch, loss, args.nSn, evalR,
                          elbo, elbo_loglik.mean(), elbo_loglik_val.mean(),
                          complexity, ent, logprior.mean(), log_jacobians.mean()))
            elbo_hist.append(elbo)

        scheduler.step(running_loss)

        if scheduler.has_convergence_been_reached():
            print('INFO: Converence has been reached. Stopping iterations.')
            break

    return resolution_network, elbo_hist


def evaluate(resolution_network, args, R):

    resolution_network.eval()

    with torch.no_grad():

        xis = sample_q(args, R, exact=True)  # [R, args.w_dim]
        xis = xis.to(args.device)
        # hmonomial = torch.prod(xis ** args.hs.T, dim=1)

        thetas, log_jacobians = resolution_network(xis)  # [R, args.w_dim], [R]

        print('thetas min {} max {}'.format(thetas.min(), thetas.max()))
        print('xis[0] point mass? {}'.format(torch.max(xis, dim=0).values[0] - torch.min(xis, dim=0).values[0]))
        print('xis min {} max {}'.format(xis.min(), xis.max()))

        args.theta_lower = torch.min(thetas, dim=0).values.detach()
        args.theta_upper = torch.max(thetas, dim=0).values.detach()

        args.xi_upper = torch.max(xis, dim=0).values.detach()
        if args.exact_EqLogq and args.method != 'nf_gammatrunc':
            ent = qj_entropy(args).sum()
        else:
            ent = q_entropy_sample(args, xis)

        complexity = ent - log_prior(args, thetas, xis).mean() - log_jacobians.mean()

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            data, target = data.to(args.device), target.to(args.device)
            elbo_loglik += loglik(thetas, data, target, args).sum(dim=1)

        elbo = elbo_loglik.mean() - complexity

        elbo_loglik_val = np.array([0.0])
        # for batch_idx, (data, target) in enumerate(args.val_loader):
        #     elbo_loglik_val += loglik(thetas, data, target, args).sum(dim=1)

        return elbo, elbo_loglik.mean(), complexity, ent, log_prior(args, thetas, xis).mean(), log_jacobians.mean(), elbo_loglik_val.mean()


# for given sample size and supposed lambda, learn resolution map g and return acheived ELBO (plus entropy)
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--data', nargs='*')
    parser.add_argument('--prior_dist', nargs='*')

    parser.add_argument('--mode', nargs='*')

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--blundell_weighting', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--trainR', type=int, default=5)
    parser.add_argument('--exact_EqLogq', action='store_true')

    parser.add_argument('--display_interval',type=int, default=100)
    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    args.dataset, args.H, args.sample_size, args.zeromean = args.data
    args.H = int(args.H)
    args.sample_size = int(args.sample_size)

    args.prior, args.prior_var = args.prior_dist
    args.prior_var = float(args.prior_var)

    get_dataset_by_id(args)
    args.batch_size = np.int(np.round(args.sample_size/10))

    if args.mode[0] == 'nf_gamma':

        args.method = 'nf_gamma'
        args.no_couplingpairs = int(args.mode[1])
        args.nf_hidden = int(args.mode[2])
        args.lmbda0 = float(args.mode[3])
        args.k0 = float(args.mode[4])
        args.betastar = args.mode[5]
        set_gengamma_varparams(args)

    elif args.mode[0] == 'nf_gaussian':

        args.method = 'nf_gaussian'
        args.no_couplingpairs = int(args.mode[1])
        args.nf_hidden = int(args.mode[2])
        args.nf_gaussian_mean = float(args.mode[3])
        args.nf_gaussian_var = float(args.mode[4])

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)

    net, elbo_hist = train(args)
    elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=100)
    elbo_val = elbo_loglik_val.mean() - complexity
    print('nSn {}, elbo {} '
          '= loglik {} (loglik_val {}) - [complexity {} = qentropy {} - logprior {} - logjacob {} ], '
          .format(args.nSn, elbo, elbo_loglik.mean(), elbo_loglik_val.mean(), complexity, ent,
                  logprior.mean(), log_jacobians.mean()))

    print('exact elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
    # print('validation: exact elbo {} plus entropy {} = {} for sample size n {}'.format(elbo_val, args.nSn_val, elbo_val+args.nSn_val, args.sample_size))
    print('-lambda log n + (m-1) log log n: {}'.format(-args.trueRLCT*np.log(args.sample_size) + (args.truem-1.0)*np.log(np.log(args.sample_size))))
    # print('true lmbda {} versus supposed lmbda {}'.format(args.trueRLCT, args.lmbda_star))

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

    # elif args.method == 'mf_gaussian':
    #
    #     # TODO: might be out of date, especially w.r.t. prior
    #     print(args)
    #     net = train_pyvarinf(args)
    #     elbo, _, _ = evaluate_pyvarinf(net, args, R=10)
    #
    #     print('exact elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
    #     print('-lambda log n + (m-1) log log n: {}'.format(
    #         -args.trueRLCT * np.log(args.sample_size) + (args.truem - 1.0) * np.log(np.log(args.sample_size))))
    #     print('true lmbda {}'.format(args.trueRLCT))

    results_dict = {'elbo': elbo, 'elbo_loglik': elbo_loglik, 'complexity': complexity,
                    'elbo_val': elbo_val,
                    'elbo_loglik_val': elbo_loglik_val,
                    'asy_log_pDn': -args.trueRLCT * np.log(args.sample_size) + (args.truem - 1.0) * np.log(np.log(args.sample_size)),
                    'elbo_hist': elbo_hist}

    if args.path is not None:
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(vars(args), '{}/args.pt'.format(args.path))
        # torch.save(net.state_dict(), '{}/state_dict.pt'.format(args.path))
        torch.save(results_dict, '{}/results.pt'.format(args.path))


if __name__ == "__main__":
    main()

