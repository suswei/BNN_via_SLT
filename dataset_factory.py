from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import math
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


class tanh_network(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, H=1):
        super(tanh_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, H, bias=False)
        self.fc2 = nn.Linear(H, output_dim, bias=False)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class reducedrank(nn.Module):
    def __init__(self, input_dim, output_dim, H):
        super(reducedrank, self).__init__()
        self.fc1 = nn.Linear(input_dim, H, bias=False)
        self.fc2 = nn.Linear(H, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_dataset_by_id(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # univariate input x, uniform [-1, 1]
    # univariate output y is normal with variance 1
    # and mean \sum_{m=1}^args.H a_m tanh(b_m x)
    if args.dataset == 'tanh':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        # model
        args.model = tanh_network(H=args.H)
        args.w_dim = 2 * args.H
        max_integer = int(math.sqrt(args.H))
        args.trueRLCT = (args.H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)
        if max_integer ** 2 == args.H:
            args.truem = 2
        else:
            args.truem = 1

        # training
        m = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        X = m.sample(torch.Size([args.sample_size]))
        if args.zeromean == 'True':
            mean = torch.zeros(args.sample_size, 1)
        else:
            if args.prior == 'gaussian':
                theta_a = torch.FloatTensor(1, args.H).normal_(mean=0, std=args.prior_var**(1/2))
                theta_b = torch.FloatTensor(1, args.H).normal_(mean=0, std=args.prior_var**(1/2))
            else:
                theta_a = torch.FloatTensor(1, args.H).uniform_(0)
                theta_b = torch.FloatTensor(1, args.H).uniform_(0)
            mean = torch.matmul(theta_a, torch.tanh(theta_b.T * X.T)).T
        y_rv = Normal(mean, 1)
        y = y_rv.sample()
        args.nSn = -y_rv.log_prob(y).sum()
        args.train_loader = torch.utils.data.DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

        # validation
        X_val = m.sample(torch.Size([args.sample_size]))
        if args.zeromean == 'True':
            mean = torch.zeros(args.sample_size, 1)
        else:
            mean = torch.matmul(theta_a, torch.tanh(theta_b.T * X_val.T)).T
        y_rv = Normal(mean, 1)
        y_val = y_rv.sample()
        args.nSn_val = -y_rv.log_prob(y_val).sum()
        args.val_loader = torch.utils.data.DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=True)

        # create smaller datasets
        ns = [args.sample_size//4, args.sample_size//3, args.sample_size//2]
        args.datasets = []
        args.ns = ns
        args.nSns = []
        for n in ns:
            X = m.sample(torch.Size([n]))
            y_rv = Normal(0.0, 1)
            y = y_rv.sample(torch.Size([n, 1]))
            args.nSns += [- y_rv.log_prob(y).sum()]
            args.datasets += [torch.utils.data.DataLoader(TensorDataset(X, y))]
        args.ns += [args.sample_size]
        args.nSns += [args.nSn]
        args.datasets += [args.train_loader]

    elif args.dataset == 'tanh_general':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        # model
        args.model = tanh_network(H=args.H)
        args.w_dim = 3 * args.H
        max_integer = int(math.sqrt(args.H))
        args.trueRLCT = args.H/2
        if max_integer ** 2 == args.H:
            args.truem = 2
        else:
            args.truem = 1

        # training
        m = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        X = m.sample(torch.Size([args.sample_size]))
        mean = torch.zeros(args.sample_size, 1)
        y_rv = Normal(mean, 1)
        y = y_rv.sample()

        # properties of data
        args.nSn = -y_rv.log_prob(y).sum()
        args.train_loader = torch.utils.data.DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

        # validation
        X_val = m.sample(torch.Size([args.sample_size]))
        mean = 0.0
        y_rv = Normal(mean, 1)
        y_val = y_rv.sample(torch.Size([args.sample_size, 1]))
        args.val_loader = torch.utils.data.DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=True)
        args.nSn_val = -y_rv.log_prob(y_val).sum()

        # create smaller datasets
        ns = [args.sample_size//4, args.sample_size//3, args.sample_size//2]
        args.datasets = []
        args.ns = ns
        args.nSns = []
        for n in ns:
            X = m.sample(torch.Size([n]))
            y_rv = Normal(0.0, 1)
            y = y_rv.sample(torch.Size([n, 1]))
            args.nSns += [- y_rv.log_prob(y).sum()]
            args.datasets += [torch.utils.data.DataLoader(TensorDataset(X, y))]
        args.ns += [args.sample_size]
        args.nSns += [args.nSn]
        args.datasets += [args.train_loader]


    # multivariate input x, Gaussian
    # multivariate output y (dim = args.H) is normal with variance 1
    # and mean BAx
    elif args.dataset == 'reducedrank':

        args.output_dim = args.H
        args.input_dim = args.output_dim + 3
        args.a_params = torch.transpose(
            torch.cat((torch.eye(args.H), torch.ones([args.H, args.input_dim - args.H], dtype=torch.float32)), 1), 0,
            1)  # input_dim * H
        args.b_params = torch.eye(args.output_dim)

        args.model = reducedrank(input_dim=args.input_dim, output_dim=args.output_dim, H=args.H)
        args.w_dim = (args.input_dim + args.output_dim) * args.H
        if args.w_dim % 2 != 0:
            print('Warning: the NF employed requires args.w_dim be even')
        args.trueRLCT = (args.output_dim * args.H - args.H ** 2 + args.input_dim * args.H) / 2  # rank r = H
        args.truem = 1

        # generate x
        m = MultivariateNormal(torch.zeros(args.input_dim), torch.eye(
            args.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
        X = m.sample(torch.Size([args.sample_size]))
        # generate y
        mean = torch.matmul(torch.matmul(X, args.a_params), args.b_params)
        y_rv = MultivariateNormal(mean, torch.eye(args.output_dim))
        y = y_rv.sample()

        # properties of data
        args.nSn = - y_rv.log_prob(y).sum()
        args.train_loader = torch.utils.data.DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

        X_val = m.sample(torch.Size([args.sample_size]))
        mean = torch.matmul(torch.matmul(X_val, args.a_params), args.b_params)
        y_rv = MultivariateNormal(mean, torch.eye(args.output_dim))
        y_val = y_rv.sample()
        args.val_loader = torch.utils.data.DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=True)
        args.nSn_val = -y_rv.log_prob(y_val).sum()

        # create smaller datasets
        # ns = get_ns(args.sample_size)
        ns = [args.sample_size//4, args.sample_size//3, args.sample_size//2]
        args.datasets = []
        args.ns = ns
        args.nSns = []
        for n in ns:
            X = m.sample(torch.Size([n]))
            mean = torch.matmul(torch.matmul(X, args.a_params), args.b_params)
            y_rv = MultivariateNormal(mean, torch.eye(args.output_dim))
            y = y_rv.sample()
            args.nSns += [- y_rv.log_prob(y).sum()]
            args.datasets += [torch.utils.data.DataLoader(TensorDataset(X, y))]
        args.ns += [args.sample_size]
        args.nSns += [args.nSn]
        args.datasets += [args.train_loader]

    else:
        print('Not a valid dataset name. See options in dataset-factory')


def get_lmbda(Hs, dataset):
    """

    :param Hs: list of hidden units values
    :param dataset: string name
    :return: list of corresponding RLCTs
    """

    trueRLCT = []
    for H in Hs:
        if dataset == 'reducedrank':
            output_dim = H
            input_dim = output_dim + 3
            trueRLCT += [(output_dim * H - H ** 2 + input_dim * H) / 2]  # rank r = H

        elif dataset=='tanh':
            max_integer = int(math.sqrt(H))
            trueRLCT += [(H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)]

    return trueRLCT


def loglik(theta, data, target, args):
    """

    :param theta: $R$ samples of theta
    :param data:
    :param target:
    :param args:
    :param R:
    :return: R by batch_size log probability matrix, 1/b \sum_{i=1}^b \log p(y_i|x_i,theta_1), ... , 1/b \sum_{i=1}^b \log p(y_i|x_i,theta_R)
    """

    if args.dataset == 'reducedrank':

        a_dim = args.a_params.shape[0] * args.a_params.shape[1]
        R = theta.shape[0]
        logprob = torch.empty(R, data.shape[0])
        for r in range(R):
            theta_a = theta[r, 0:a_dim].reshape(args.a_params.shape[0], args.a_params.shape[1])
            theta_b = theta[r, a_dim:].reshape(args.b_params.shape[0], args.b_params.shape[1])

            mean = torch.matmul(torch.matmul(data, theta_a), theta_b)

            y_rv = MultivariateNormal(mean, torch.eye(args.output_dim))
            logprob[r, :] = y_rv.log_prob(target)

    elif args.dataset == 'tanh':

        R = theta.shape[0]
        B = data.shape[0]

        theta_a = theta[:, 0:args.H]  # R by H
        theta_b = theta[:, args.H:]  # R by H
        means = torch.empty(R, B)
        for r in range(R):
            # 1 by B
            means[r,] = torch.matmul(theta_a[r,].unsqueeze(dim=1).T, torch.tanh(theta_b[r,].unsqueeze(dim=1) * data.T))
        y_rv = MultivariateNormal(means.unsqueeze(dim=2), torch.eye(1))
        logprob = y_rv.log_prob(target.repeat(1, theta.shape[0]).T.unsqueeze(dim=2))

    elif args.dataset == 'tanh_general':
        R = theta.shape[0]
        B = data.shape[0]

        theta_a = theta[:, 0:args.H]  # R by H
        theta_b = theta[:, args.H:2*args.H]  # R by H
        theta_c = theta[:,2*args.H:]
        means = torch.empty(R, B)
        for r in range(R):
            # 1 by B
            means[r,] = torch.matmul(theta_a[r,].unsqueeze(dim=1).T, torch.tanh(theta_b[r,].unsqueeze(dim=1) * data.T+theta_c[r,].unsqueeze(dim=1)))
        y_rv = MultivariateNormal(means.unsqueeze(dim=2), torch.eye(1))
        logprob = y_rv.log_prob(target.repeat(1, theta.shape[0]).T.unsqueeze(dim=2))

    return logprob  # R by B
