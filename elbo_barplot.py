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

    parser.add_argument('--path', type=str, help='where the desingularized mean field results are located')

    parser.add_argument('--dataset', type=str, choices=['reducedrank', 'tanh'])

    parser.add_argument('--Hs', nargs="+", type=int) # currently hardcoded for len(Hs)=3

    parser.add_argument('--savefig', action='store_true')

    parser.add_argument('--lmbda_grid', nargs="+", type=int, default=None)

    args = parser.parse_args()

    # if args.savefig:
    #     matplotlib.use("pgf")
    #     matplotlib.rcParams.update({
    #         "pgf.texsystem": "xelatex",
    #         'font.family': 'serif',
    #         'text.usetex': True,
    #         'pgf.rcfonts': False,
    #     })
    #     plt.rcParams["figure.figsize"] = (6.75/2, 3)
    sns.set_style("ticks")

    # if args.lmbda_grid is None:
    #     args.lmbda_grid = get_lmbda(args.Hs, args.dataset)
    #
    # tuned_lmbdas = get_best_lmbda(args)

    for n in [5000]:

        for prior_var in [1e-1, 1e-2]:

            Hs_list = []
            method_list = []
            seed_list = []
            ev_list = []

            for i in range(len(args.Hs)):

                H = args.Hs[i]

                for seed in [1, 2, 3, 4, 5]:

                    for method in ['nf_gammatrunc','nf_gamma','truth']:

                        if method == 'truth':

                            path = '{}/{}_{}_n{}_H{}_prior{}_seed{}_varparams{}'.format(args.path, 'nf_gammatrunc',
                                                                                        args.dataset, n, H, prior_var,
                                                                                        seed, varparams)
                            ev_list += [torch.load('{}/results.pt'.format(path))['asy_log_pDn']]
                            method_list += [method]
                            seed_list += ['{}'.format(seed)]
                            Hs_list += [H]

                        else:
                            for varparams in ['allones', 'icml', 'abs_gauss_n','exp']:

                                Hs_list += [H]
                                method_list += ['{}_{}'.format(method,varparams)]
                                seed_list += ['{}'.format(seed)]

                                path = '{}/{}_{}_n{}_H{}_prior{}_seed{}_varparams{}'.format(args.path,method, args.dataset, n, H, prior_var, seed, varparams)
                                ev_list += [torch.load('{}/results.pt'.format(path))['elbo'].detach().numpy()
                                            + torch.load('{}/args.pt'.format(path))['nSn'].numpy()]

                                # elbo_hist_nf_gammatrunc = torch.load('{}/results.pt'.format(path))['elbo_hist']
                                # elbo_hist_nf_gaussian = torch.load('{}/results.pt'.format(path))['elbo_hist']

                                # plt.plot(elbo_hist_nf_gaussian,'r',label='gaussian')
                                # plt.plot(elbo_hist_nf_gammatrunc,label='gammatrunc')
                                # plt.title('dataset {} H {} seed {}'.format(args.dataset, H, seed))
                                # plt.show()

            method_list = pd.Series(method_list, dtype='category')
            seed_list = pd.Series(seed_list, dtype="category")

            summary_pd = pd.DataFrame({'H': Hs_list,
                                       'ELBOplusnSn': ev_list,
                                       'method': method_list,
                                       'seed': seed_list})

            g = sns.barplot(x="H", y="ELBOplusnSn",
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

            plt.title('gaussian prior var {}, n {}'.format(prior_var, n))

            if args.savefig:
                # plt.savefig('{}/{}_n{}_prior{}.pgf'.format(args.path, args.dataset, n, prior_var), bbox_inches='tight')
                plt.savefig('{}/{}_n{}_prior{}.png'.format(args.path, args.dataset, n, prior_var), bbox_inches='tight')

            else:
                plt.show()
            plt.close()


if __name__ == "__main__":
    main()

