import torch
import numpy as np
from torch.distributions.normal import Normal
from dataset_factory import loglik
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd


def plot_pred_dist(thetas, args, saveimgpath):

    # f = plt.figure(figsize=(6, 4))
    # ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
    # ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    f.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)

    # plot the drawn parameters
    if args.H == 1:
        ax1.scatter(thetas[:, 0], thetas[:, 1], c='lightblue', s=20, label='drawn parameters')
        ax1.plot(args.theta_a.squeeze(dim=0), args.theta_b.squeeze(dim=0), 'r*', label='true')
        ax1.set_xlabel(r'$\theta_1$')
        ax1.set_ylabel(r'$\theta_2$')
        ax1.legend()

    else:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(thetas)
        pca_pd = pd.DataFrame({'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1]})
        # sns.scatterplot(ax=ax1,
        #     x="PC1", y="PC2",
        #     palette='lightblue',
        #     data=pca_pd,
        #     alpha=0.3
        # )
        ax1.scatter(pca_result[:, 0], pca_result[:, 1], c='lightblue', s=20, label='drawn parameters')
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')

    ax1.set_title('drawn parameters')

    if args.H == 1:
        xl = -1.0
        xu = 1.0
    else:
        xl = 0.25
        xu = 1.0
    tempx = torch.arange(xl, xu, .01)
    predictions = torch.empty(size=(thetas.shape[0], tempx.shape[0]))
    i = 0
    for theta in thetas:
        ll, mean = loglik(theta.unsqueeze(dim=0), tempx.unsqueeze(dim=1), tempx.unsqueeze(dim=1), args)
        y_rv = Normal(mean, 1)
        y = y_rv.sample()
        predictions[i] = y
        i += 1

    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)
    # plot 90% confidence level of predictions
    ax2.fill_between(
        tempx, percentiles[0, :], percentiles[1, :], color="lightblue", label='90% band'
    )

    ll, true_mean = loglik(torch.hstack((args.theta_a, args.theta_b)), tempx.unsqueeze(dim=1), tempx.unsqueeze(dim=1), args)
    ax2.plot(tempx, true_mean.squeeze(dim=0), 'r-', linewidth=1, label='truth')
    # also show the original data points
    ax2.plot(args.X[args.X >= xl ], args.y[args.X >= xl ], 'ko', markersize=1, label='data points')
    ax2.set_xlim(xl, xu)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.legend()
    ax2.set_title('approximate Bayes posterior predictive')

    plt.savefig(saveimgpath)