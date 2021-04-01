import seaborn as sns
import torch
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import argparse

from dataset_factory import get_lmbda
from sweep import set_sweep_config


def get_best_lmbda(args):

    tuned_lmbdas = []
    sweep_params, _ = set_sweep_config()

    seeds = sweep_params['seeds']

    for H in args.Hs:
        lmbda_H = np.empty([len(args.lmbda_grid),len(seeds)])
        i = 0
        for lmbda in args.lmbda_grid:

            j = 0
            for seed in seeds:
                path = '{}/transformed_{}_H{}_lmbda{}_seed{}'.format(args.transformed_path, args.dataset, H, round(lmbda), seed)
                lmbda_H[i,j] = torch.load('{}/results.pt'.format(path))['elbo'].detach().numpy()
                j += 1

            i +=1

        tuned_lmbdas += [args.lmbda_grid[lmbda_H.mean(1).argmax()]]

    return tuned_lmbdas


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--transformed_path', type=str, help='where the desingularized mean field results are located')

    parser.add_argument('--dataset', type=str, choices=['reducedrank','tanh'])

    parser.add_argument('--Hs', nargs="+", type=int) # currently hardcoded for len(Hs)=3

    parser.add_argument('--savefig', action='store_true')

    parser.add_argument('--lmbda_grid', nargs="+", type=int, default=None)

    args = parser.parse_args()

    if args.savefig:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "xelatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
        plt.rcParams["figure.figsize"] = (6.75/2, 3)

    Hs_list =[]
    method_list =[]
    seed_list = []
    ev_list = []

    if args.lmbda_grid is None:
        args.lmbda_grid = get_lmbda(args.Hs, args.dataset)

    tuned_lmbdas = get_best_lmbda(args)

    for i in range(len(args.Hs)):

        H = args.Hs[i]

        for method in ['DMF', 'GMF', 'truth']:

            for seed in [1, 2, 3, 4, 5]:

                Hs_list += ['{}'.format(H)]
                method_list += [method]
                seed_list += ['{}'.format(seed)]

                if method == 'DMF':

                    lmbda = round(tuned_lmbdas[i])
                    path = '{}/transformed_{}_H{}_lmbda{}_seed{}'.format(args.transformed_path, args.dataset, H, lmbda, seed)
                    ev_list += [torch.load('{}/results.pt'.format(path))['elbo'].detach().numpy()
                                + torch.load('{}/args.pt'.format(path))['nSn'].numpy()]

                elif method == 'GMF':

                    path = 'untransformed2/untransformed_{}_H{}_seed{}'.format(args.dataset, H, seed)
                    ev_list += [torch.load('{}/results.pt'.format(path))['elbo'].detach().numpy()
                                + torch.load('{}/args.pt'.format(path))['nSn'].numpy()]

                elif method == 'truth':

                    oracle_lmbda = round(tuned_lmbdas[i])
                    path = '{}/transformed_{}_H{}_lmbda{}_seed{}'.format(args.transformed_path, args.dataset, H, oracle_lmbda, seed)
                    ev_list += [torch.load('{}/results.pt'.format(path))['asy_log_pDn']]

    method_list = pd.Series(method_list, dtype="category")
    seed_list = pd.Series(seed_list, dtype="category")

    summary_pd = pd.DataFrame({'$H$': Hs_list,
                               'ELBO $+ nS_n$': ev_list,
                               'method': method_list,
                               'seed': seed_list})
    sns.set_style("ticks")
    g = sns.barplot(x="$H$", y="ELBO $+ nS_n$",
                    hue="method",
                    data=summary_pd)
    hatches = ['/', '/', '/',
               '+', '+', '+',
               'x', 'x', 'x']
    for hatch, patch in zip(hatches, g.patches):
        patch.set_hatch(hatch)
    leg = plt.legend(bbox_to_anchor=(1, 1), loc=2)
    for patch in leg.get_patches():
        patch.set_height(12)
        patch.set_y(-6)

    if args.savefig:
        plt.savefig('{}_{}.pgf'.format(args.transformed_path, args.dataset), bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()