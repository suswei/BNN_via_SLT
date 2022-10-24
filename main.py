import os
import argparse
from dataset_factory import *
from normalizing_flows import *
from utils import *


def train(args):

    resolution_network = RealNVP(args.nf_couplingpair, args.nf_hidden, args.w_dim, args.sample_size, args.device, args.grad_flag)
    print(resolution_network)
    params = list(resolution_network.named_parameters())
    def is_varparam(n):
        return 'lmbdas' in n or 'ks' in n or 'betas' in n

    # convergence is quite sensitive to variational parameter learning rates
    args.lr_lmbda = args.lr*100
    args.lr_k = args.lr
    args.lr_beta = args.lr*100

    grouped_parameters = [
        {"params": [p for n, p in params if 'lmbdas' in n], 'lr': args.lr_lmbda},
        {"params": [p for n, p in params if 'ks' in n], 'lr': args.lr_k},
        {"params": [p for n, p in params if 'betas' in n], 'lr': args.lr_beta},
        {"params": [p for n, p in params if not is_varparam(n)], 'lr': args.lr},
    ]

    resolution_network.to(args.device)

    optimizer = torch.optim.Adam(grouped_parameters, lr=args.lr)
    torch.autograd.set_detect_anomaly(True)

    elbo_hist = []
    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            data, target = data.to(args.device), target.to(args.device)

            resolution_network.train()
            optimizer.zero_grad()

            xis = resolution_network.sample_xis(args.trainR, args.base_dist, upper=args.upper)  # [R, args.w_dim]

            thetas, log_jacobians = resolution_network(xis)  # log_jacobians [R, 1]  E_q log |g'(xi)|
            if torch.any(torch.isnan(thetas)):
                print('nan thetas')

            args.theta_lower = torch.min(thetas, dim=0).values.detach()
            args.theta_upper = torch.max(thetas, dim=0).values.detach()

            loglik_elbo_vec, _ = loglik(thetas, data, target, args)  # [R, minibatch_size] E_q \sum_i=1^m p(y_i |x_i , g(\xi))
            complexity = Eqj_logqj(resolution_network, args).sum() - log_prior(args, thetas).mean() - log_jacobians.mean()  # q_entropy no optimization
            elbo = loglik_elbo_vec.mean(dim=0).sum() - complexity * (args.batch_size / args.sample_size)

            running_loss += -elbo.item()

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:

            evalR = 100
            elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val \
                = evaluate(resolution_network, args, R=evalR)
            print('epoch {}: loss {}, nSn {}, (R = {}) elbo {} '
                  '= loglik {} (loglik_val {}) - [complexity {} = Eqlogq {} - logprior {} - logjacob {} ], '
                  .format(epoch, loss, args.nSn, evalR,
                          elbo, elbo_loglik.mean(), elbo_loglik_val.mean(),
                          complexity, ent, logprior.mean(), log_jacobians.mean()))
            elbo_hist.append(elbo)

    return resolution_network, elbo_hist


def evaluate(resolution_network, args, R):

    resolution_network.eval()

    with torch.no_grad():

        xis = resolution_network.sample_xis(R, args.base_dist, upper=args.upper)# [R, args.w_dim]
        thetas, log_jacobians = resolution_network(xis)  # [R, args.w_dim], [R]

        print('thetas min {} max {}'.format(thetas.min(), thetas.max()))
        # print('xis[0] point mass? {}'.format(torch.max(xis, dim=0).values[0] - torch.min(xis, dim=0).values[0]))
        print('xis min {} max {}'.format(xis.min(), xis.max()))

        args.theta_lower = torch.min(thetas, dim=0).values.detach()
        args.theta_upper = torch.max(thetas, dim=0).values.detach()

        args.xi_upper = torch.max(xis, dim=0).values.detach()

        ent = Eqj_logqj(resolution_network, args).sum()
        complexity = ent - log_prior(args, thetas).mean() - log_jacobians.mean()

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            data, target = data.to(args.device), target.to(args.device)
            temp, _ = loglik(thetas, data, target, args)
            temp = temp.sum(dim=1)
            elbo_loglik += temp

        elbo = elbo_loglik.mean() - complexity

        elbo_loglik_val = np.array([0.0])
        # for batch_idx, (data, target) in enumerate(args.val_loader):
        #     elbo_loglik_val += loglik(thetas, data, target, args).sum(dim=1)

        return elbo, elbo_loglik.mean(), complexity, ent, log_prior(args, thetas).mean(), log_jacobians.mean(), elbo_loglik_val.mean()


# for given sample size and supposed lambda, learn resolution map g and return acheived ELBO (plus entropy)
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--seeds', nargs='*', default=[1])

    parser.add_argument('--data', nargs='*', default=['tanh', 16, 5000, True],
                        help='[0]: tanh or rr '
                             '[1]: H '
                             '[2]: sample size '
                             '[3]: zeromean')

    parser.add_argument('--prior_dist', nargs='*', default=['gaussian', 0, 1])

    parser.add_argument('--var_mode', nargs='*', default=['gengammatrunc', 2, 16],
                        help='[0]: gengamma or gengammatrunc or gaussian'
                             '[1]: nf_couplingpair'
                             '[2]: nf_hidden')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--trainR', type=int, default=5)
    parser.add_argument('--grad_flag', type=bool, default=True)

    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--path', type=str)
    parser.add_argument('--viz', action='store_true')

    args = parser.parse_args()

    for seed in args.seeds:

        args.seed = int(seed)

        args.dataset, args.H, args.sample_size, args.zeromean = args.data
        args.H = int(args.H)
        args.sample_size = int(args.sample_size)

        # TODO: needs to take into account other prior options in utils.py
        args.prior, args.prior_mean, args.prior_var = args.prior_dist
        args.prior_mean = float(args.prior_mean)
        args.prior_var = float(args.prior_var)

        get_dataset_by_id(args)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(args)

        args.base_dist = args.var_mode[0]
        args.nf_couplingpair = int(args.var_mode[1])
        args.nf_hidden = int(args.var_mode[2])

        if args.base_dist == 'gengamma' or args.base_dist == 'gengammatrunc':

            args.upper = 1 # should be input for gengammatrunc

        elif args.base_dist == 'gaussian':

            if len(args.var_mode) == 3:
                args.nf_gaussian_mean = 0.0
                args.nf_gaussian_var = 1.0
            else:
                args.nf_gaussian_mean = float(args.var_mode[3])
                args.nf_gaussian_var = float(args.var_mode[4])

        net, elbo_hist = train(args)
        elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=100)
        elbo_val = elbo_loglik_val.mean() - complexity
        print('nSn {}, elbo {} = loglik {} (loglik_val {}) - [complexity {} = Eq_j log q_j {} - logprior {} - logjacob {} ]'
              .format(args.nSn, elbo, elbo_loglik.mean(), elbo_loglik_val.mean(), complexity, ent, logprior.mean(), log_jacobians.mean()))
        print('elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
        print('-lambda log n + (m-1) log log n: {}'.format(-args.trueRLCT*np.log(args.sample_size) + (args.truem-1.0)*np.log(np.log(args.sample_size))))

        results_dict = {'elbo': elbo,
                        'elbo_loglik': elbo_loglik,
                        'complexity': complexity,
                        'elbo_val': elbo_val,
                        'elbo_loglik_val': elbo_loglik_val,
                        'asy_log_pDn': -args.trueRLCT * np.log(args.sample_size) + (args.truem - 1.0) * np.log(np.log(args.sample_size)),
                        'elbo_hist': elbo_hist}

        if args.path is not None:
            path = '{}/seed{}'.format(args.path, args.seed)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(vars(args), '{}/args.pt'.format(path))
            # torch.save(net.state_dict(), '{}/state_dict.pt'.format(args.path))
            torch.save(results_dict, '{}/results.pt'.format(path))

        if args.dataset == 'tanh' and args.viz:

            from plot_pred_dist import plot_pred_dist

            net.eval()
            with torch.no_grad():
                xis = net.sample_xis(500, args.base_dist, upper=args.upper)  # [R, args.w_dim]
                thetas, log_jacobians = net(xis)

            l = len(args.data) + len(args.prior_dist) + len(args.var_mode)
            if args.base_dist == 'gengamma':
                args.var_mode[4] = float(args.var_mode[4]).as_integer_ratio()
            saveimgpath = 'output/'+('{}_'*l).format(*args.data, *args.prior_dist, *args.var_mode, args.grad_flag) + 'epoch{}_pred_dist'.format(args.epochs)
            print(saveimgpath)
            plot_pred_dist(thetas, args, saveimgpath)

            # TOD0: shouldn't hard code, pass in from dataset_factory
            if args.zeromean=='True':
                w0 = 0
            else:
                w0 = 5

            if args.dataset == 'tanh':
                with open('{}.tex'.format(saveimgpath), 'w') as file:
                    file.write('The model of interest is the tanh network with hidden units $H={}$. '
                               'The data is generated according to $p_0(y | x, w) = p(y | x, w_0)$ where $w_0 = {}$. '
                               'The prior is taken to be $N({}, {} I_d)$. '
                               'The predictive distributions resulting from different variational approximations trained on a dataset of size $n={}$ are displayed.'
                               .format(args.H, w0, args.prior_mean, args.prior_var, args.sample_size))
                    # file.write('$H=${}'.format(args.H))


if __name__ == "__main__":
    main()

