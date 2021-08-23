import os
import argparse
from dataset_factory import *
import custom_lr_scheduler
from normalizing_flows import *
from utils import *


# TODO: alternate training lambda, k, beta and NF params?
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

    resolution_network = RealNVP(nets, nett, masks, args.w_dim, args.trueRLCT)
    params = list(resolution_network.named_parameters())

    def is_varparam(n):
        return 'lambdas' in n or 'ks' in n

    def is_varparam2(n):
        return 'lmbdas' in n or 'ks' in n or 'betas' in n

    grouped_parameters = [
        {"params": [p for n, p in params if is_varparam(n)], 'lr': args.lr * 10}, #TODO: a more systematic way to set these lr's
        {"params": [p for n, p in params if 'lambdas' in n], 'lr': args.lr * 100},
        {"params": [p for n, p in params if not is_varparam2(n)], 'lr': args.lr},
    ]

    resolution_network.to(args.device)

    optimizer = torch.optim.Adam(grouped_parameters, lr=args.lr)
    scheduler = custom_lr_scheduler.CustomReduceLROnPlateau(optimizer)

    elbo_hist = []
    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            data, target = data.to(args.device), target.to(args.device)

            resolution_network.train()
            optimizer.zero_grad()

            xis = sample_q(resolution_network, args, args.trainR)  # [R, args.w_dim]
            xis = xis.to(args.device)

            thetas, log_jacobians = resolution_network(xis)  # log_jacobians [R, 1]  E_q log |g'(xi)|

            args.theta_lower = torch.min(thetas, dim=0).values.detach()
            args.theta_upper = torch.max(thetas, dim=0).values.detach()

            loglik_elbo_vec = loglik(thetas, data, target, args)  # [R, minibatch_size] E_q \sum_i=1^m p(y_i |x_i , g(\xi))
            complexity = exp_logqj(resolution_network, args).sum() - log_prior(args, thetas).mean() - log_jacobians.mean()  # q_entropy no optimization
            elbo = loglik_elbo_vec.mean(dim=0).sum() - complexity * (args.batch_size / args.sample_size)

            running_loss += -elbo.item()

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:

            evalR = 10
            elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val \
                = evaluate(resolution_network, args, R=evalR, exact=True)
            print('epoch {}: loss {}, nSn {}, (R = {}) exact elbo {} '
                  '= loglik {} (loglik_val {}) - [complexity {} = Eqlogq {} - logprior {} - logjacob {} ], '
                  .format(epoch, loss, args.nSn, evalR,
                          elbo, elbo_loglik.mean(), elbo_loglik_val.mean(),
                          complexity, ent, logprior.mean(), log_jacobians.mean()))
            elbo_hist.append(elbo)


        # scheduler.step(running_loss)
        #
        # if scheduler.has_convergence_been_reached():
        #     print('INFO: Converence has been reached. Stopping iterations.')
        #     break

    return resolution_network, elbo_hist


def evaluate(resolution_network, args, R, exact):

    resolution_network.eval()

    with torch.no_grad():

        xis = sample_q(resolution_network, args, R, exact=exact)  # [R, args.w_dim]
        xis = xis.to(args.device)

        thetas, log_jacobians = resolution_network(xis)  # [R, args.w_dim], [R]

        print('thetas min {} max {}'.format(thetas.min(), thetas.max()))
        # print('xis[0] point mass? {}'.format(torch.max(xis, dim=0).values[0] - torch.min(xis, dim=0).values[0]))
        print('xis min {} max {}'.format(xis.min(), xis.max()))

        args.theta_lower = torch.min(thetas, dim=0).values.detach()
        args.theta_upper = torch.max(thetas, dim=0).values.detach()

        args.xi_upper = torch.max(xis, dim=0).values.detach()

        ent = exp_logqj(resolution_network, args).sum()
        complexity = ent - log_prior(args, thetas).mean() - log_jacobians.mean()

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            data, target = data.to(args.device), target.to(args.device)
            elbo_loglik += loglik(thetas, data, target, args).sum(dim=1)

        elbo = elbo_loglik.mean() - complexity

        elbo_loglik_val = np.array([0.0])
        # for batch_idx, (data, target) in enumerate(args.val_loader):
        #     elbo_loglik_val += loglik(thetas, data, target, args).sum(dim=1)

        return elbo, elbo_loglik.mean(), complexity, ent, log_prior(args, thetas).mean(), log_jacobians.mean(), elbo_loglik_val.mean()


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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--trainR', type=int, default=5)

    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    args.dataset, args.H, args.sample_size, args.zeromean = args.data
    args.H = int(args.H)
    args.sample_size = int(args.sample_size)

    # TODO: needs to take into account other prior options in utils.py
    args.prior, args.prior_var = args.prior_dist
    args.prior_var = float(args.prior_var)
    # args.prior = args.prior_dist[0]

    get_dataset_by_id(args)
    args.batch_size = np.int(np.round(args.sample_size/10))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)

    if args.mode[0] == 'nf_gamma':

        args.method = 'nf_gamma'
        args.no_couplingpairs = int(args.mode[1])
        args.nf_hidden = int(args.mode[2])

    elif args.mode[0] == 'nf_gaussian':

        args.method = 'nf_gaussian'
        args.no_couplingpairs = int(args.mode[1])
        args.nf_hidden = int(args.mode[2])
        if len(args.mode) == 3:
            args.nf_gaussian_mean = 0.0
            args.nf_gaussian_var = 1.0
        else:
            args.nf_gaussian_mean = float(args.mode[3])
            args.nf_gaussian_var = float(args.mode[4])

    net, elbo_hist = train(args)
    print('lmbdas {}'.format(net.lmbdas))
    print('ks {}'.format(net.ks))
    print('betas {}'.format(net.betas))

    elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=100)
    elbo_val = elbo_loglik_val.mean() - complexity
    print('nSn {}, elbo {} '
          '= loglik {} (loglik_val {}) - [complexity {} = Eq_j log q_j {} - logprior {} - logjacob {} ], '
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

    results_dict = {'elbo': elbo,
                    'elbo_loglik': elbo_loglik,
                    'complexity': complexity,
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

