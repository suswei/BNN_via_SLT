import torch
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset


def load_P(model, H, device):

    if model == 'tanh':
        return OneLayerTanh(H, device)
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
        z_a = w[:, 0:self.H]  # L by H
        z_b = w[:, self.H:]  # L by H
        means = torch.empty(L, B, device=self.device)
        for l in range(L):
            means[l,] = torch.matmul(z_a[l,].unsqueeze(dim=1).T, torch.tanh(z_b[l,].unsqueeze(dim=1) * x.T))
        means = means.to(self.device)
        y_rv = MultivariateNormal(means.unsqueeze(dim=2), torch.eye(1).to(self.device))
        log_p = y_rv.log_prob(y.repeat(1, w.shape[0]).T.unsqueeze(dim=2))
        return log_p