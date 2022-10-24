import torch
import pyvarinf
from torch.distributions.normal import Normal


def train_pyvarinf(args):

    var_model = pyvarinf.Variationalize(args.model, prior_std=args.prior_var)
    optimizer = torch.optim.Adam(var_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs):

        var_model.train()
        for batch_idx, (data, target) in enumerate(args.train_loader):

            optimizer.zero_grad()
            output = var_model(data)
            loss_error = ((output-target)**2).mean()
            loss_prior = var_model.prior_loss() / args.sample_size
            loss = loss_error + loss_prior
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            elbo, elbo1, elbo2 = evaluate_pyvarinf(var_model, args, R=1)
            print('epoch {}: elbo {} = loglik {} - prior {}'.format(epoch, elbo, elbo1, elbo2))
    return var_model


def loglik_pyvarinf(var_model, train_loader):
    sample = pyvarinf.Sample(var_model=var_model)
    sample.draw()  # var_model(data) predicts with a new w drawn from q
    logprob = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = var_model(data)
        y_rv = Normal(output, 1)
        logprob += y_rv.log_prob(target).sum()

    return logprob


def evaluate_pyvarinf(var_model, args, R):

    var_model.eval()
    elbo_loglik = 0.0
    for r in range(R):
        elbo_loglik += loglik_pyvarinf(var_model, args.train_loader)
    elbo1 = elbo_loglik / R
    elbo2 = var_model.prior_loss()
    final_elbo = elbo1 - elbo2
    return final_elbo, elbo1, elbo2


