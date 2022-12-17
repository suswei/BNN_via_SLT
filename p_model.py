import torch
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset


def load_P(model, H, device):

    if model == 'tanh':
        return OneLayerTanh(H, device)
    elif model == 'reducedrank':
        return ReducedRank(H, device)
    raise NotImplementedError('Model %s not valid.' % model)


class OneLayerTanh():
    def __init__(self, H, device):

        self.H = H

        max_integer = int(H**(1/2))
        self.trueRLCT = (H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)
        if max_integer ** 2 == H:
            self.truem = 2
        else:
            self.truem = 1

        self.w_dim = 2 * H
        self.device = device

    def load_data(self, sample_size, batch_size):
        m = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
        X = m.sample(torch.Size([sample_size]))
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


class ReducedRank():
    def __init__(self, H, device):
        self.H = H
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
        m = MultivariateNormal(torch.zeros(self.input_dim), torch.eye(
            self.input_dim))  # the input_dim=output_dim + 3, output_dim = H (the number of hidden units)
        X = m.sample(torch.Size([sample_size]))
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