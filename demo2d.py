import torch
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy.stats as st
from dataset_factory import loglik


def posterior_viz(thetas, args, saveimgpath):

    wmin = thetas.detach().numpy().min()
    wmax = thetas.detach().numpy().max()
    wmin = -3
    wmax = 3
    wspace = 5
    wrange = np.linspace(wmin, wmax, wspace)
    w = np.repeat(wrange[:, None], wspace, axis=1)
    w = np.concatenate([[w.flatten()], [w.T.flatten()]])

    # prior
    logprior = -(w ** 2).sum(axis=0) / (2*args.prior_var) - np.log(2*np.pi)/2 - np.log(args.prior_var)

    # true unnormalised posterior at inverse temp
    logpost = torch.zeros(wspace * wspace)
    for i in range(wspace * wspace):
        current_w = torch.from_numpy(w[:,i]).float().unsqueeze(dim=0) # convert to torch tensor of shape [1,w_dim]
        for batch_idx, (data, target) in enumerate(args.train_loader):
            data, target = data.to(args.device), target.to(args.device)
            logpost[i] += loglik(current_w, data, target, args).sum(dim=1).squeeze()

    logpost = logpost.detach().numpy() + logprior

    # kde sampled_weights
    kernel = st.gaussian_kde([thetas[:,0].detach().numpy(), thetas[:,1].detach().numpy()])
    f = np.reshape(kernel(w).T, [wspace,wspace])

    fig = make_subplots(rows=1, cols=3, subplot_titles=('(unnormalised) prior',
                                                        '(unnormalised) posterior',
                                                        'kde of sampled weights'))
    fig.add_trace(
        go.Contour(
            z=np.exp(logprior.reshape(wspace, wspace)),
            x=wrange,  # horizontal axis
            y=wrange
        ),  # vertical axis
        row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            z=np.exp(logpost.reshape(wspace, wspace).T),
            x=wrange,  # horizontal axis
            y=wrange
        ),  # vertical axis
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=thetas[:, 0].detach().numpy(),
            y=thetas[:, 1].detach().numpy(),
            mode='markers',
            name='Sampled Data',
            marker=dict(size=3,opacity=0.6)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Contour(
            z=f.T,
            x=wrange,  # horizontal axis
            y=wrange),  # vertical axis
            row=1, col=3
    )
    fig.update_layout(title_text='{}, {}'.format(args.data, args.mode))
    # if args.notebook:
    #     fig.show(renderer='notebook')
    # else:
    #     fig.show()
    # fig.show()
    fig.show(renderer="browser")

    if saveimgpath is not None:
        fig.write_image('./{}/posteior'.format(saveimgpath))