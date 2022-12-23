import os
import argparse
from q_model import *
from p_model import *
from utils import *


def train(args, writer=None):

    resolution_network = RealNVP(args.base_dist, args.nf_couplingpair, args.nf_hidden,
                                 args.w_dim, args.sample_size, args.device, args.grad_flag)
    params = list(resolution_network.named_parameters())
    def is_varparam(n):
        return 'lmbdas' in n or 'ks' in n or 'betas' in n or 'mu' in n or 'log_sigma' in n

    args.lr_lmbda = args.lr*100
    args.lr_k = args.lr*10
    args.lr_beta = args.lr*100

    args.lr_mu = args.lr*100
    args.lr_log_sigma = args.lr*100

    grouped_parameters = [
        {"params": [p for n, p in params if 'lmbdas' in n], 'lr': args.lr_lmbda},
        {"params": [p for n, p in params if 'ks' in n], 'lr': args.lr_k},
        {"params": [p for n, p in params if 'betas' in n], 'lr': args.lr_beta},
        {"params": [p for n, p in params if 'mu' in n], 'lr': args.lr_mu},
        {"params": [p for n, p in params if 'log_sigma' in n], 'lr': args.lr_log_sigma},
        {"params": [p for n, p in params if not is_varparam(n)], 'lr': args.lr},
    ]

    resolution_network.to(args.device)

    optimizer = torch.optim.Adam(grouped_parameters)
    torch.autograd.set_detect_anomaly(True)

    the_last_loss = 1e10
    args.patience = 10
    trigger_times = 0

    for epoch in range(1, args.epochs):

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(args.train_loader):

            data, target = data.to(args.device), target.to(args.device)

            resolution_network.train()
            optimizer.zero_grad()

            xis = resolution_network.sample_xis(args.trainR, args.base_dist, upper=args.upper)  # [R, args.w_dim]

            ws, log_jacobians = resolution_network(xis)  # log_jacobians [R, 1]  E_q log |g'(xi)|
            if torch.any(torch.isnan(ws)):
                print('nan ws')

            # TODO: this code block has similar repetitions
            loglik_elbo_vec = args.P.loglik(data, target, ws)  # [R, minibatch_size] E_q \sum_i=1^m p(y_i |x_i , g(\xi))
            complexity = Eqj_logqj(resolution_network, args).sum() - log_prior(args, ws).mean() - log_jacobians.mean()  # q_entropy no optimization
            elbo = loglik_elbo_vec.mean(dim=0).sum() - complexity * (args.batch_size / args.sample_size)

            running_loss += -elbo.item()

            loss = -elbo
            loss.backward()
            optimizer.step()

            # Early stopping based on elbo on train set (not sure val set makes sense)
            # TODO: early stopping on train or validation set?
            the_current_loss = -elbo
            # print('The current loss:', the_current_loss)

            if the_current_loss > the_last_loss:
                trigger_times += 1
                # print('trigger times:', trigger_times)

                if trigger_times >= args.patience:
                    print('Early stopping!\nStart to test process.')
                    # TODO: Shouldn’t we load previously best model instead of most recent model?
                    return resolution_network

            else:
                # print('trigger times: 0')
                trigger_times = 0

            the_last_loss = the_current_loss

        if epoch % args.display_interval == 0:

            elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val, logpred \
                = evaluate(resolution_network, args, R=10)
            print('epoch {}: elbo {}, nSn {}, logpred {} '
                  .format(epoch, elbo, args.nSn, logpred))



            if args.tensorboard:
                writer.add_scalar('elbo', elbo.detach().cpu().numpy(), epoch)
                writer.add_scalar('elbo_loglik_val', elbo_loglik_val.detach().cpu().numpy(), epoch)

    return resolution_network


def evaluate(resolution_network, args, R):

    resolution_network.eval()

    with torch.no_grad():

        xis = resolution_network.sample_xis(R, args.base_dist, upper=args.upper)# [R, args.w_dim]
        ws, log_jacobians = resolution_network(xis)  # [R, args.w_dim], [R]

        print('ws min {} max {}'.format(ws.min(), ws.max()))
        # print('xis[0] point mass? {}'.format(torch.max(xis, dim=0).values[0] - torch.min(xis, dim=0).values[0]))
        print('xis min {} max {}'.format(xis.min(), xis.max()))

        ent = Eqj_logqj(resolution_network, args).sum()
        complexity = ent - log_prior(args, ws).mean() - log_jacobians.mean()

        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            data, target = data.to(args.device), target.to(args.device)
            elbo_loglik += args.P.loglik(data, target, ws).sum(dim=1)

        elbo_loglik_val = 0.0
        logpred = 0.0
        for batch_idx, (data, target) in enumerate(args.val_loader):
            data, target = data.to(args.device), target.to(args.device)
            loglik = args.P.loglik(data, target, ws) # temp.shape = [number of ws, sample size of data]
            logpred += torch.log(torch.exp(loglik).mean(dim=0)).sum(dim=0)
            elbo_loglik_val += loglik.sum(dim=1)

        logpred = logpred/args.val_size
        elbo = elbo_loglik.mean() - complexity

        return elbo, elbo_loglik.mean(), complexity, ent, log_prior(args, ws).mean(), log_jacobians.mean(), elbo_loglik_val.mean(), logpred


# for given sample size and supposed lambda, learn resolution map g and return acheived ELBO (plus entropy)
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--seeds', nargs='*', default=[1])

    parser.add_argument('--data', nargs='*', default=['tanh', 16, 5000],
                        help='[0]: tanh or rr '
                             '[1]: H '
                             '[2]: sample size ')

    parser.add_argument('--prior_dist', nargs='*', default=['gaussian', 0, 1])

    parser.add_argument('--var_mode', nargs='*', default=['gengamma', 2, 16, True],
                        help='[0]: gengamma or gengammatrunc or gaussian_std or gaussian_match'
                             '[1]: nf_couplingpair'
                             '[2]: nf_hidden'
                             '[3]: grad_flag')

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--trainR', type=int, default=5)

    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--path', type=str)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--tensorboard', action='store_true')

    args = parser.parse_args()

    # parse args.data
    args.dataset, args.H, args.sample_size = args.data
    args.H = int(args.H)
    args.sample_size = int(args.sample_size)
    print('only full batch supported, setting batch size to sample size')
    args.batch_size = args.sample_size

    # parse args.prior_dist
    # TODO: needs to take into account other prior options in utils.py
    args.prior, args.prior_mean, args.prior_var = args.prior_dist
    args.prior_mean = float(args.prior_mean)
    args.prior_var = float(args.prior_var)

    # parse args.var_mode
    args.base_dist = args.var_mode[0]
    args.nf_couplingpair = int(args.var_mode[1])
    args.nf_hidden = int(args.var_mode[2])
    args.grad_flag = args.var_mode[3] == 'True'

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)

    for seed in args.seeds:

        args.seed = int(seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        args_str = '{}_{}_seed{}'.format(args.data, args.var_mode, args.seed)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter('tensorboard/{}'.format(args_str))
        else:
            writer=None

        args.P = load_P(args.dataset, args.H, args.device)
        args.train_loader, args.nSn = args.P.load_data(args.sample_size, args.sample_size)
        args.val_size = 1000
        args.val_loader, _ = args.P.load_data(args.val_size, args.val_size)
        args.w_dim = args.P.w_dim
        args.trueRLCT = args.P.trueRLCT
        args.upper = 1

        net = train(args, writer)

        elbo, elbo_loglik, complexity, ent, logprior, log_jacobians, elbo_loglik_val, logpred = evaluate(net, args, R=1000)
        if args.tensorboard:
            writer.add_scalar('elbo', elbo.detach().cpu().numpy(), args.epochs)
            # writer.add_scalar('predloglik', predloglik.mean(), args.epochs) # this is a popular diagnostic but it's very problematic as it could have nothing to do with the optimization objective

        print('nSn {}, (R = {}) elbo {} = loglik {} - [complexity {} = Eq_j log q_j {} - logprior {} - logjacob {} ]'
              .format(args.nSn, evalR, elbo, elbo_loglik.mean(), complexity, ent, logprior.mean(), log_jacobians.mean()))
        print('elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
        if args.P.trueRLCT is not None:
            asy_log_pDn = -args.P.trueRLCT*np.log(args.sample_size) + (args.P.truem-1.0)*np.log(np.log(args.sample_size))
            print('-lambda log n + (m-1) log log n: {}'.format(asy_log_pDn))
        else:
            asy_log_pDn = - args.P.w_dim/2 * np.log(args.sample_size)
            print('-d/2 log n: {}'.format(asy_log_pDn))
        results_dict = {'elbo': elbo,
                        'elbo_loglik': elbo_loglik,
                        'complexity': complexity,
                        'predloglik': logpred,
                        'asy_log_pDn': asy_log_pDn}

        if args.path is not None:
            path = '{}/seed{}'.format(args.path, args.seed)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(vars(args), '{}/args.pt'.format(path))
            # torch.save(net.state_dict(), '{}/state_dict.pt'.format(args.path))
            torch.save(results_dict, '{}/results.pt'.format(path))


if __name__ == "__main__":
    main()

