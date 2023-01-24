import os
import argparse
from q_model import *
from p_model import *
from utils import *
from scipy.special import logsumexp


def train(args, P, writer=None):

    resolution_network = RealNVP(args.base_dist, args.nf_couplingpair, args.nf_hidden,
                                 args.w_dim, args.sample_size, args.device, args.grad_flag)
    params = list(resolution_network.named_parameters())

    args.qdim = sum(p.numel() for p in resolution_network.parameters() if p.requires_grad)
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

    for epoch in range(1, args.epochs):

        for batch_idx, (data, target) in enumerate(args.train_loader):

            data, target = data.to(args.device), target.to(args.device)

            resolution_network.train()
            optimizer.zero_grad()

            xis = resolution_network.sample_xis(args.trainR, args.base_dist, upper=args.upper)  # [R, args.w_dim]
            ws, q_log_prob = resolution_network.log_prob(xis)
            elbo_complexity = q_log_prob - P.logprior(ws).mean()
            elbo_loglik = P.loglik(data, target, ws).mean(dim=0).sum()  # [R, minibatch_size] E_q \sum_i=1^m p(y_i |x_i , g(\xi))
            elbo = elbo_loglik - elbo_complexity * (args.batch_size / args.sample_size)

            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % args.display_interval == 0:

            elbo, test_lpd \
                = evaluate_elbo_testlpd(resolution_network, P, args, R=10)
            print('epoch {}: elbo {}, nSn {}, test_lpd {}, vge {}'
                  .format(epoch, elbo, args.nSn, test_lpd, -args.nSn_val/args.val_size - test_lpd))


            if args.tensorboard:
                writer.add_scalar('elbo', elbo.detach().cpu().numpy(), epoch)

    return resolution_network


def evaluate_elbo_testlpd(resolution_network, P, args, R):

    resolution_network.eval()
    with torch.no_grad():

        xis = resolution_network.sample_xis(R, args.base_dist, upper=args.upper)  # [R, args.w_dim]
        ws, q_log_prob = resolution_network.log_prob(xis)

        elbo_complexity = q_log_prob - P.logprior(ws).mean()
        elbo_loglik = 0.0
        for batch_idx, (data, target) in enumerate(args.train_loader):
            data, target = data.to(args.device), target.to(args.device)
            elbo_loglik += P.loglik(data, target, ws).sum(dim=1)
        elbo = elbo_loglik.mean() - elbo_complexity

        test_lpd = 0.0
        for batch_idx, (data, target) in enumerate(args.val_loader):
            data, target = data.to(args.device), target.to(args.device)
            loglik = P.loglik(data, target, ws) # temp.shape = [number of ws, sample size of data]
            test_lpd += logsumexp(loglik.detach().cpu().numpy(), axis=0, b=1.0/loglik.shape[0]).sum()
        test_lpd = test_lpd/args.val_size

        return elbo, test_lpd


def estimate_nSn(args):

    P = load_P(args.dataset, args.H, args.device, args.prior_mean, args.prior_var, True)
    early_stopper = EarlyStopper(patience=5, min_delta=1.0)
    optimizer = torch.optim.Adam(P.parameters(), lr=0.001)
    for epoch in range(0, 2000):
        P.train()
        for batch_idx, (data, target) in enumerate(args.train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            loss = -P.loglik_w1_w2(data, target, P.w1.to(args.device), P.w2.to(args.device)).mean()
            loss.backward()
            optimizer.step()

        P.eval()
        with torch.no_grad():
            for batch_idx, (val_data, val_target) in enumerate(args.val_loader):
                validation_loss = -P.loglik_w1_w2(val_data.to(args.device), val_target.to(args.device), P.w1.to(args.device), P.w2.to(args.device)).mean()
            if early_stopper.early_stop(validation_loss):
                break

        if epoch % 100 == 0 or epoch == 1999:
            print('epoch {}: train loss {}, validation loss {}, patience {}'.format(epoch, loss, validation_loss, early_stopper.counter))
            estimated_nSn = -P.loglik_w1_w2(data, target, P.w1.to(args.device), P.w2.to(args.device)).sum()
            print('epoch {}: estmated nSn {}, true nSn {}'.format(epoch, estimated_nSn, args.nSn))

    return estimated_nSn


# for given sample size and supposed lambda, learn resolution map g and return acheived ELBO (plus entropy)
def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--seed', type=int, default=1985)
    parser.add_argument('--data', nargs='*', default=['tanh', 16],
                        help='[0]: tanh or rr '
                             '[1]: H ')
    parser.add_argument('--ns', nargs='*', default=[1000])
    parser.add_argument('--prior_dist', nargs='*', default=[0, 1], help='only supports Gaussian, pass in mean and variance')

    parser.add_argument('--var_mode', nargs='*', default=['gengamma', 2, 16, True],
                        help='[0]: gengamma or gengammatrunc or gaussian or gaussian_match'
                             '[1]: nf_couplingpair'
                             '[2]: nf_hidden'
                             '[3]: grad_flag')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--trainR', type=int, default=10)
    parser.add_argument('--estimate_entropy', action='store_true')

    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--path', type=str)
    parser.add_argument('--tensorboard', action='store_true')

    args = parser.parse_args()


    # parse args.prior_dist
    args.prior_mean, args.prior_var = args.prior_dist
    args.prior_mean = float(args.prior_mean)
    args.prior_var = float(args.prior_var)

    # parse args.var_mode
    args.base_dist = args.var_mode[0]
    args.nf_couplingpair = int(args.var_mode[1])
    args.nf_hidden = int(args.var_mode[2])
    args.grad_flag = args.var_mode[3] == 'True'

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.dataset, args.H = args.data
    args.H = int(args.H)

    for n in args.ns:

        # parse args.data
        print('only full batch supported, setting batch size to sample size')
        args.sample_size = int(n)
        args.batch_size = args.sample_size

        args_str = '{}_n{}_{}_seed{}'.format(args.dataset, args.sample_size, args.var_mode, args.seed)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter('tensorboard/{}'.format(args_str))
        else:
            writer=None

        P = load_P(args.dataset, args.H, args.device, args.prior_mean, args.prior_var, False)
        args.train_loader, args.nSn = P.load_data(args.sample_size, args.sample_size)
        args.val_size = 10000
        args.val_loader, args.nSn_val = P.load_data(args.val_size, args.val_size)
        args.w_dim = P.w_dim
        args.trueRLCT = P.trueRLCT
        args.upper = 1

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        net = train(args, P, writer)

        print(args)

        elbo, test_lpd = evaluate_elbo_testlpd(net, P, args, R=1000)
        if args.estimate_entropy:
            args.estimated_nSn = estimate_nSn(args)
        else:
            args.estimated_nSn = None

        if args.tensorboard:
            writer.add_scalar('elbo', elbo.detach().cpu().numpy(), args.epochs)
            writer.add_scalar('test_lpd', test_lpd.mean(), args.epochs)

        print('elbo {} plus entropy {} = {} for sample size n {}'.format(elbo, args.nSn, elbo+args.nSn, args.sample_size))
        print('vge {}'.format(-args.nSn_val/args.val_size - test_lpd))
        if P.trueRLCT is not None:
            asy_log_pDn = -P.trueRLCT*np.log(args.sample_size) + (P.truem-1.0)*np.log(np.log(args.sample_size))
            print('-lambda log n + (m-1) log log n: {}'.format(asy_log_pDn))
        else:
            asy_log_pDn = - P.w_dim/2 * np.log(args.sample_size)
            print('-d/2 log n: {}'.format(asy_log_pDn))
        results_dict = {'elbo': elbo,
                        'test_lpd': test_lpd,
                        'asy_log_pDn': asy_log_pDn}

        path = '{}/n{}_seed{}'.format(args.path, args.sample_size, args.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        args_dict = vars(args)
        del args_dict['train_loader']
        del args_dict['val_loader']
        torch.save(args_dict, '{}/args.pt'.format(path))
        torch.save(results_dict, '{}/results.pt'.format(path))


if __name__ == "__main__":
    main()

