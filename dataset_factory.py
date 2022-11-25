from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import math
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


def get_dataset_by_id(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # univariate input x, uniform [-1, 1]
    # univariate output y is normal with variance 1
    # and mean \sum_{m=1}^args.H a_m tanh(b_m x)
    if args.dataset == 'tanh':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        # model
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
        if args.zeromean=='True':
            w_a = torch.zeros(1, args.H)
            w_b = torch.zeros(1, args.H)
            mean = torch.zeros(args.sample_size, 1)
        else:
            w_a = 5*torch.ones(1, args.H)
            w_b = 5*torch.ones(1, args.H)
            mean = torch.matmul(w_a, torch.tanh(w_b.T * X.T)).T

        y_rv = Normal(mean, 1)
        y = y_rv.sample()
        args.nSn = -y_rv.log_prob(y).sum()
        args.train_loader = torch.utils.data.DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

        # validation
        X_val = m.sample(torch.Size([args.sample_size]))
        if args.zeromean=='True':
            mean = torch.zeros(args.sample_size, 1)
        else:
            mean = torch.matmul(w_a, torch.tanh(w_b.T * X_val.T)).T
        y_rv = Normal(mean, 1)
        y_val = y_rv.sample()
        args.nSn_val = -y_rv.log_prob(y_val).sum()
        args.val_loader = torch.utils.data.DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=True)

        # create smaller datasets
        ns = [int(round(np.exp(4))) * 32, int(round(np.exp(5))) * 32, int(round(np.exp(6))) * 32,
              int(round(np.exp(7))) * 32]
        args.datasets = []
        args.Xs = []
        args.Ys = []
        args.ns = ns
        args.nSns = []
        for n in ns:
            X = m.sample(torch.Size([n]))
            y_rv = Normal(0.0, 1)
            y = y_rv.sample(torch.Size([n, 1]))
            args.Xs += [X]
            args.Ys += [y]
            args.nSns += [- y_rv.log_prob(y).sum()]
            args.datasets += [torch.utils.data.DataLoader(TensorDataset(X, y))]
        args.ns += [args.sample_size]
        args.nSns += [args.nSn]
        args.datasets += [args.train_loader]

    elif args.dataset == 'tanh_general':  # "Resolution of Singularities ... for Layered Neural Network" Aoyagi and Watanabe

        # model
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

        args.w_dim = (args.input_dim + args.output_dim) * args.H
        if args.w_dim % 2 != 0:
            print('Warning: the NF employed requires args.w_dim be even')
        args.trueRLCT = (args.output_dim * args.H - args.H ** 2 + args.input_dim * args.H) / 2  # rank r = H
        args.truem = 1 # TODO: theoretical support for this?

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


    else:
        print('Not a valid dataset name. See options in dataset-factory')

    return X, y, X_val, y_val


def get_lmbda_dim(Hs, dataset):
    """

    :param Hs: list of hidden units values
    :param dataset: string name
    :return: list of corresponding RLCTs
    """

    trueRLCT = []
    dim = []
    for H in Hs:
        if dataset == 'reducedrank':
            output_dim = H
            input_dim = output_dim + 3
            trueRLCT += [(output_dim * H - H ** 2 + input_dim * H) / 2]  # rank r = H
            dim += [input_dim * H + output_dim * H]

        elif dataset=='tanh':
            max_integer = int(math.sqrt(H))
            trueRLCT += [(H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)]
            dim += [2*H]

    return trueRLCT, dim


def loglik(w, data, target, args):
    """

    :param w: R samples of w
    :param data:
    :param target:
    :param args:
    :return: R by batch_size log probability matrix, 1/b \sum_{i=1}^b \log p(y_i|x_i,w_1), ... , 1/b \sum_{i=1}^b \log p(y_i|x_i,w_R)
    """

    if args.dataset == 'reducedrank':

        a_dim = args.a_params.shape[0] * args.a_params.shape[1]
        R = w.shape[0]
        logprob = torch.empty(R, data.shape[0])
        for r in range(R):
            w_a = w[r, 0:a_dim].reshape(args.a_params.shape[0], args.a_params.shape[1])
            w_b = w[r, a_dim:].reshape(args.b_params.shape[0], args.b_params.shape[1])

            mean = torch.matmul(torch.matmul(data, w_a), w_b)
            mean = mean.to(args.device)

            y_rv = MultivariateNormal(mean, torch.eye(args.output_dim).to(args.device))
            logprob[r, :] = y_rv.log_prob(target)
        logprob = logprob.to(args.device)
        means = []
    elif args.dataset == 'tanh':

        R = w.shape[0]
        B = data.shape[0]

        w_a = w[:, 0:args.H]  # R by H
        w_b = w[:, args.H:]  # R by H
        means = torch.empty(R, B)
        for r in range(R):
            # 1 by B
            means[r,] = torch.matmul(w_a[r,].unsqueeze(dim=1).T, torch.tanh(w_b[r,].unsqueeze(dim=1) * data.T))
        means = means.to(args.device)
        y_rv = MultivariateNormal(means.unsqueeze(dim=2), torch.eye(1).to(args.device))
        logprob = y_rv.log_prob(target.repeat(1, w.shape[0]).T.unsqueeze(dim=2))


    elif args.dataset == 'tanh_general':
        R = w.shape[0]
        B = data.shape[0]

        w_a = w[:, 0:args.H]  # R by H
        w_b = w[:, args.H:2*args.H]  # R by H
        w_c = w[:,2*args.H:]
        means = torch.empty(R, B)
        for r in range(R):
            # 1 by B
            means[r,] = torch.matmul(w_a[r,].unsqueeze(dim=1).T, torch.tanh(w_b[r,].unsqueeze(dim=1) * data.T+w_c[r,].unsqueeze(dim=1)))
        y_rv = MultivariateNormal(means.unsqueeze(dim=2), torch.eye(1).to(args.device))
        logprob = y_rv.log_prob(target.repeat(1, w.shape[0]).T.unsqueeze(dim=2))

    return logprob, means  # R by B
