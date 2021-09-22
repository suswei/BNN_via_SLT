import torch
import numpy as np
from torch.distributions.normal import Normal
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import scipy.stats as st
from dataset_factory import loglik
from matplotlib import pyplot as plt


# def posterior_viz(thetas, args, saveimgpath):
# 
#     wmin = thetas.detach().numpy().min()
#     wmax = thetas.detach().numpy().max()
#     wmin = -3
#     wmax = 3
#     wspace = 5
#     wrange = np.linspace(wmin, wmax, wspace)
#     w = np.repeat(wrange[:, None], wspace, axis=1)
#     w = np.concatenate([[w.flatten()], [w.T.flatten()]])
#
#     # prior
#     logprior = -(w ** 2).sum(axis=0) / (2*args.prior_var) - np.log(2*np.pi)/2 - np.log(args.prior_var)
#
#     # true unnormalised posterior at inverse temp
#     logpost = torch.zeros(wspace * wspace)
#     for i in range(wspace * wspace):
#         current_w = torch.from_numpy(w[:,i]).float().unsqueeze(dim=0) # convert to torch tensor of shape [1,w_dim]
#         for batch_idx, (data, target) in enumerate(args.train_loader):
#             data, target = data.to(args.device), target.to(args.device)
#             temp, _ = loglik(current_w, data, target, args)
#             logpost[i] += temp.sum(dim=1).squeeze()
#
#     logpost = logpost.detach().numpy() + logprior
#
#     # kde sampled_weights
#     kernel = st.gaussian_kde([thetas[:,0].detach().numpy(), thetas[:,1].detach().numpy()])
#     f = np.reshape(kernel(w).T, [wspace,wspace])
#
#     fig = make_subplots(rows=1, cols=3, subplot_titles=('(unnormalised) prior',
#                                                         '(unnormalised) posterior',
#                                                         'kde of sampled weights'))
#     fig.add_trace(
#         go.Contour(
#             z=np.exp(logprior.reshape(wspace, wspace)),
#             x=wrange,  # horizontal axis
#             y=wrange
#         ),  # vertical axis
#         row=1, col=1
#     )
#     fig.add_trace(
#         go.Contour(
#             z=np.exp(logpost.reshape(wspace, wspace).T),
#             x=wrange,  # horizontal axis
#             y=wrange
#         ),  # vertical axis
#         row=1, col=2
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=thetas[:, 0].detach().numpy(),
#             y=thetas[:, 1].detach().numpy(),
#             mode='markers',
#             name='Sampled Data',
#             marker=dict(size=3,opacity=0.6)
#         ),
#         row=1, col=2
#     )
#     fig.add_trace(
#         go.Contour(
#             z=f.T,
#             x=wrange,  # horizontal axis
#             y=wrange),  # vertical axis
#             row=1, col=3
#     )
#     fig.update_layout(title_text='{}, {}'.format(args.data, args.mode))
#     # if args.notebook:
#     #     fig.show(renderer='notebook')
#     # else:
#     #     fig.show()
#     # fig.show()
#     fig.show(renderer="browser")
#
#     # if saveimgpath is not None:
#     #     fig.write_image('{}'.format(saveimgpath))


def pred_dist(thetas, args, saveimgpath):

    f = plt.figure(figsize=(6, 4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    # f,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))

    # plot the drawn parameters
    ax1.scatter(thetas[:, 0], thetas[:, 1], c='lightblue', s=20, label='drawn parameters')
    ax1.plot(args.theta_a.squeeze(dim=0), args.theta_b.squeeze(dim=0), 'r*', label='true')
    ax1.legend()
    ax1.set_title('drawn parameters')

    tempx = torch.range(-1, 1, .01)
    predictions = torch.empty(size=(thetas.shape[0], tempx.shape[0]))
    i=0
    for theta in thetas:
        ll, mean = loglik(theta.unsqueeze(dim=0), tempx.unsqueeze(dim=1), tempx.unsqueeze(dim=1), args)
        y_rv = Normal(mean, 1)
        y = y_rv.sample()
        predictions[i] = y
        i+=1

    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)
    # plot 90% confidence level of predictions
    ax2.fill_between(
        tempx, percentiles[0, :], percentiles[1, :], color="lightblue", label='90% band'
    )

    ll, true_mean = loglik(torch.hstack((args.theta_a, args.theta_b)), tempx.unsqueeze(dim=1), tempx.unsqueeze(dim=1), args)
    ax2.plot(tempx, true_mean.squeeze(dim=0), 'r-', linewidth=1, label='truth')
    # also show the original data points
    ax2.plot(args.X, args.y, 'ko', markersize=1, label='data points')
    ax2.set_xlim(-1, 1)
    ax2.legend()
    ax2.set_title('approximate Bayes posterior predictive')

    plt.savefig(saveimgpath)