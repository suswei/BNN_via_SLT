import os
import argparse
from dataset_factory import *
import custom_lr_scheduler
from normalizing_flows import *
from utils import *
from scipy import stats


def train(args):

    if args.nf == 'iaf':
        resolution_network = IAF(latent_size=args.w_dim, h_size=args.w_dim)
    elif args.nf == 'rnvp':
        resolution_network = RealNVP(dim=args.w_dim, hidden_dim=args.nf_hidden, layers=args.nf_layers, af=args.nf_af)
    elif args.nf == 'vanilla_rnvp':
        # resolution_network = R_NVP(d=args.w_dim, K0net = args.K0net)
        nets = lambda: nn.Sequential(nn.Linear(args.w_dim, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(),
                                     nn.Linear(256, args.w_dim), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(args.w_dim, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(),
                                     nn.Linear(256, args.w_dim))

        # masks = (torch.eye(args.w_dim).reshape(args.w_dim, args.w_dim)).repeat(no_layers, 1)

        ones = np.ones(args.w_dim)
        ones[np.random.choice(args.w_dim, args.w_dim // 2)] = 0
        half_mask = torch.cat((torch.from_numpy(ones.astype(np.float32)).unsqueeze(dim=0), torch.from_numpy((1-ones).astype(np.float32)).unsqueeze(dim=0) ))

        if args.K0net == 'True':
            masks = half_mask.repeat(args.nf_layers-1, 1)
            ones = np.ones(args.w_dim)
            ones[0] = 0
            masks = torch.cat((masks, torch.from_numpy(ones.astype(np.float32)).unsqueeze(dim=0)))
        else:
            masks = half_mask.repeat(args.nf_layers, 1)

        resolution_network = RealNVP(nets, nett, masks, args.w_dim)

    optimizer = torch.optim.Adam(resolution_network.parameters(), lr=args.lr)
    scheduler = custom_lr_scheduler.CustomReduceLROnPlateau\
        (optimizer, 'min', verbose=True, factor=0.9, patience=100, eps=1e-6)

    elbo_hist = []
    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            resolution_network.train()
            optimizer.zero_grad()

            if args.prior == 'unif':
                args.trainR = 10

            xis = sample_q(args, args.trainR, exact=True)  # [R, args.w_dim]

            thetas, log_jacobians = resolution_network(xis)  # log_jacobians [R, 1]  E_q log |g'(xi)|
            args.theta_lower = torch.min(thetas, dim=0).values.detach()
            args.theta_upper = torch.max(thetas, dim=0).values.detach()

            loglik_elbo_vec = loglik(thetas, data, target, args)  # [R, minibatch_size] E_q \sum_i=1^m p(y_i |x_i , g(\xi))

            complexity = - log_prior(args, thetas).mean() - log_jacobians.mean()  # q_entropy no optimization

            if args.blundell_weighting:
                M = args.sample_size/args.batch_size # number of minibatches
                pi = (2**(M-batch_idx))/(2**M-1) # follows blundell, is bad for nf_gammatrunc
                elbo = loglik_elbo_vec.mean(dim=0).sum() - complexity*pi
            else:
                elbo = loglik_elbo_vec.mean(dim=0).sum() - complexity * (args.batch_size / args.sample_size)
                # elbo = loglik_elbo_vec.mean(dim=0).sum() + (torch.prod(xis,dim=1)*args.batch_size/np.sqrt(args.sample_size)) - complexity * (args.batch_size / args.sample_size)

            running_loss += -elbo.item()

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:
            evalR = 1
            elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val \
                = evaluate(resolution_network, args, R=evalR)
            print('epoch {}: loss {}, nSn {}, (R = {}) elbo {} '
                  '= loglik {} (loglik_val {}) - [complexity {} = qentropy {} - logprior {} - logjacob {}], '
                  .format(epoch, loss, args.nSn, evalR, elbo, elbo_loglik.mean(), elbo_loglik_val.mean(), complexity, ent, logprior.mean(), log_jacobians.mean()))
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
        thetas, log_jacobians = resolution_network(xis)  # [R, args.w_dim], [R]

        print('thetas min {} max {}'.format(thetas.min(), thetas.max()))
        print('xis min {} max {}'.format(xis.min(), xis.max()))

        args.theta_lower = torch.min(thetas, dim=0).values.detach()
        args.theta_upper = torch.max(thetas, dim=0).values.detach()

        args.xi_upper = torch.max(xis, dim=0).values.detach()
        if args.exact_EqLogq and args.method != 'nf_gammatrunc':
            ent = qj_entropy(args).sum()
        else:
            ent = q_entropy_sample(args, xis)

        complexity = ent - log_prior(args, thetas).mean() - log_jacobians.mean()

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            elbo_loglik += loglik(thetas, data, target, args).sum(dim=1)

        elbo = elbo_loglik.mean() - complexity

        elbo_loglik_val = 0.0
        for batch_idx, (data, target) in enumerate(args.val_loader):
            elbo_loglik_val += loglik(thetas, data, target, args).sum(dim=1)

        ktheta = (elbo_loglik + args.nSn)/args.sample_size
        monomial = torch.prod(xis**(2*args.ks.T), dim=1)
        print('resolution map quality {}'.format(((ktheta-monomial)**2).sum()))

    return elbo, elbo_loglik.mean(), complexity, ent, log_prior(args, thetas).mean(), log_jacobians.mean(), elbo_loglik_val.mean()


# for given sample size and supposed lambda, learn resolution map g and return acheived ELBO (plus entropy)
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--dataset', type=str, default='tanh',
                        help='dataset name from dataset_factory.py (default: )',
                        choices=['reducedrank', 'tanh','tanh_general'])
    parser.add_argument('--zeromean', type=str, default='True')
    parser.add_argument('--H', type=int, default=1)

    parser.add_argument('--sample_size', type=int, default=5000,
                        help='sample size of synthetic dataset')

    parser.add_argument('--prior', type=str, default='gaussian')
    # parser.add_argument('--prior_var', type=float)

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--blundell_weighting', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--trainR', type=int, default = 1)

    parser.add_argument('--nf', type=str,default='rnvp', choices=['iaf','rnvp','vanilla_rnvp'])
    parser.add_argument('--nf_hidden', type=int, default=16)
    parser.add_argument('--nf_layers', type=int, default=20)
    parser.add_argument('--nf_af', type=str, default='relu',choices=['relu','tanh'])
    parser.add_argument('--K0net', type=str, default='False', choices=['True','False'])

    parser.add_argument('--method', type=str, default='nf_gamma', choices=['nf_gamma','nf_gammatrunc','nf_gaussian','mf_gaussian'])
    parser.add_argument('--nf_gamma_mode', type=str, default='icml')
    parser.add_argument('--lmbda_star', action='store_true')
    parser.add_argument('--beta_star', action='store_true')
    parser.add_argument('--exact_EqLogq', action='store_true')

    parser.add_argument('--display_interval',type=int, default=100)
    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    get_dataset_by_id(args)
    args.prior_var = 1/args.H

    print(args)

    if args.method == 'nf_gamma' or args.method == 'nf_gammatrunc' or args.method == 'nf_gaussian':

        if args.nf_gamma_mode == 'abs_gauss':

            args.lmbdas = 0.5*torch.ones(args.w_dim, 1)
            args.ks = torch.ones(args.w_dim, 1)
            args.betas = 0.5*torch.ones(args.w_dim, 1)

        elif args.nf_gamma_mode == 'exp':

            args.lmbdas = torch.ones(args.w_dim, 1)
            args.ks = 0.5*torch.ones(args.w_dim, 1)
            args.betas = torch.ones(args.w_dim, 1)

        elif args.nf_gamma_mode == 'icml':

            args.lmbdas = args.trueRLCT*torch.ones(args.w_dim, 1)
            args.ks = torch.ones(args.w_dim, 1)
            args.betas = args.trueRLCT*torch.ones(args.w_dim, 1)

        elif args.nf_gamma_mode == 'allones':

            args.lmbdas = torch.ones(args.w_dim, 1)
            args.ks = torch.ones(args.w_dim, 1)
            args.betas = torch.ones(args.w_dim, 1)

        if args.lmbda_star:
            args.lmbdas[0] = args.trueRLCT
        if args.beta_star:
            args.betas[0] = args.sample_size

        args.hs = args.lmbdas * 2 * args.ks - 1

    net, elbo_hist = train(args)
    elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val = evaluate(net, args, R=100)
    elbo_val = elbo_loglik_val.mean() - complexity
    print('nSn {}, elbo {} '
          '= loglik {} (loglik_val {}) - [complexity {} = qentropy {} - logprior {} - logjacob {}], '
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
        torch.save(net.state_dict(), '{}/state_dict.pt'.format(args.path))
        torch.save(results_dict, '{}/results.pt'.format(args.path))


if __name__ == "__main__":
    main()

