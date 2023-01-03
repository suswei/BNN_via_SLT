import torch
import torch.distributions as ttd
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset
from torch import nn
from torch.nn.functional import relu
import numpy as np


def load_P(model, H, device, prior_mean, prior_var):

    if model == 'tanh_zeromean':
        return OneLayerTanh(H, device, prior_mean, prior_var, True)
    elif model == 'tanh':
        return OneLayerTanh(H, device, prior_mean, prior_var, False)
    elif model == 'reducedrank':
        return ReducedRank(H, device, prior_mean, prior_var)
    elif model == 'ffrelu':
        return FFReLU(H, device, prior_mean, prior_var)
    raise NotImplementedError('Model %s not valid.' % model)


class OneLayerTanh():
    def __init__(self, H, device, prior_mean, prior_var, zeromean):

        self.H = H
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.zeromean = zeromean

        if self.zeromean:
            max_integer = int(H**(1/2))
            self.trueRLCT = (H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)
            if max_integer ** 2 == H:
                self.truem = 2
            else:
                self.truem = 1
        else:
            self.trueRLCT = None
            self.truem = None

        self.w_dim = 2 * H
        self.device = device

    def load_data(self, sample_size, batch_size):
        X_dist = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        X = X_dist.sample(torch.Size([sample_size]))
        if self.zeromean:
            a = torch.zeros(self.H, 1)
            b = torch.zeros(self.H, 1)
        else:
            a = torch.randn(self.H, 1)
            b = torch.randn(self.H, 1)

        means = torch.matmul(a.T, torch.tanh(b * X.T))
        y_rv = Normal(means.T, 1.0)
        Y = y_rv.sample()
        loader = torch.utils.data.DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        nSn = -y_rv.log_prob(Y).sum()
        return loader, nSn

    def loglik(self, x, y, w):
        L = w.shape[0]
        B = x.shape[0]
        w_a = w[:, 0:self.H]  # L by H
        w_b = w[:, self.H:]  # L by H
        means = torch.empty(L, B, device=self.device)
        for l in range(L):
            means[l, ] = torch.matmul(w_a[l, ].unsqueeze(dim=1).T, torch.tanh(w_b[l, ].unsqueeze(dim=1) * x.T))
        means = means.to(self.device)
        y_rv = Normal(means.unsqueeze(dim=2), 1.0)
        log_p = y_rv.log_prob(y.repeat(1, w.shape[0]).T.unsqueeze(dim=2))

        return log_p
    def logprior(self, ws):
        return - self.w_dim/2*np.log(2*np.pi) \
               - (1/2)*self.w_dim*np.log(self.prior_var) \
               - torch.diag(torch.matmul(ws-self.prior_mean, (ws-self.prior_mean).T))/(2*self.prior_var)

class ReducedRank():
    def __init__(self, H, device, prior_mean, prior_var):
        self.H = H
        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.output_dim = H
        self.input_dim = self.output_dim + 3
        self.a_params = torch.transpose(
            torch.cat((torch.eye(H), torch.ones([H, self.input_dim - H], dtype=torch.float32)), 1), 0,
            1)  # input_dim * H
        self.b_params = torch.eye(self.output_dim)

        self.trueRLCT = (self.output_dim * H - H ** 2 + self.input_dim * H) / 2  # rank r = H
        self.truem = 1
        
        self.w_dim = (self.input_dim + self.output_dim) * H
        if self.w_dim % 2 != 0:
            print('Warning: the NF employed requires args.w_dim be even')
        self.device = device

    def load_data(self, sample_size, batch_size):
        # generate x
        X_dist = MultivariateNormal(torch.zeros(self.input_dim), torch.eye(
            self.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
        X = X_dist.sample(torch.Size([sample_size]))
        # generate y
        mean = torch.matmul(torch.matmul(X, self.a_params), self.b_params)
        y_rv = MultivariateNormal(mean, torch.eye(self.output_dim))
        Y = y_rv.sample()
        
        loader = torch.utils.data.DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        nSn = -y_rv.log_prob(Y).sum()
        return loader, nSn

    def loglik(self, x, y, w):

        a_dim = self.a_params.shape[0] * self.a_params.shape[1]
        R = w.shape[0]
        logprob = torch.empty(R, x.shape[0])
        for r in range(R):
            w_a = w[r, 0:a_dim].reshape(self.a_params.shape[0], self.a_params.shape[1])
            w_b = w[r, a_dim:].reshape(self.b_params.shape[0], self.b_params.shape[1])

            mean = torch.matmul(torch.matmul(x, w_a), w_b)
            mean = mean.to(self.device)

            y_rv = MultivariateNormal(mean, torch.eye(self.output_dim).to(self.device))
            logprob[r, :] = y_rv.log_prob(y)
        
        log_p = logprob.to(self.device)

        return log_p

    def logprior(self, ws):
        return - self.w_dim/2*np.log(2*np.pi) \
               - (1/2)*self.w_dim*np.log(self.prior_var) \
               - torch.diag(torch.matmul(ws-self.prior_mean, (ws-self.prior_mean).T))/(2*self.prior_var)


class FFReLU():

    def __init__(self, H, device, prior_mean, prior_var):
        self.H = H
        self.prior_mean = prior_mean
        self.prior_var = prior_var

        self.input_dim = 13
        self.output_dim = 1

        self.trueRLCT = None
        self.truem = None

        self.w_dim = (self.input_dim + self.output_dim) * H
        self.device = device

    def load_data(self, sample_size, batch_size):

        # generate x
        m = MultivariateNormal(torch.zeros(self.input_dim), torch.eye(
            self.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
        X = m.sample(torch.Size([sample_size]))

        # generate y
        w1 = ttd.Normal(0, 1).sample((self.H, self.input_dim))
        means = torch.relu(w1 @ X.T)  # number of samples of w * sample size of X
        w2 = ttd.Normal(0, 1).sample((self.output_dim, self.H))
        means = w2 @ means  # number of samples of w * sample size of X
        y_rv = ttd.Normal(means.T, 1.0)
        Y = y_rv.sample()

        loader = torch.utils.data.DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        nSn = -y_rv.log_prob(Y).sum()
        return loader, nSn

    def loglik(self, x, y, w):

        w1_dim = self.H * self.input_dim
        R = w.shape[0]
        logprob = torch.empty(R, x.shape[0])
        for r in range(R):
            w1 = w[r, 0:w1_dim].reshape(self.H, self.input_dim)
            w2 = w[r, w1_dim:].reshape(self.output_dim, self.H)
            means = torch.relu(w1 @ x.T)
            means = w2 @ means
            means = means.to(self.device) # number of samples of w * sample size of X

            y_rv = ttd.Normal(means.T, 1.0)
            logprob[r, :] = y_rv.log_prob(y).T

        log_p = logprob.to(self.device)

        return log_p

    def logprior(self, ws):
        return - self.w_dim/2*np.log(2*np.pi) \
               - (1/2)*self.w_dim*np.log(self.prior_var) \
               - torch.diag(torch.matmul(ws-self.prior_mean, (ws-self.prior_mean).T))/(2*self.prior_var)