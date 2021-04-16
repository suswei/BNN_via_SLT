import seaborn as sns
import torch
import pandas as pd
from matplotlib import pyplot as plt
import argparse


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--path', type=str)

    parser.add_argument('--dataset', type=str, choices=['reducedrank', 'tanh'])

    parser.add_argument('--Hs', nargs="+", type=int) # currently hardcoded for len(Hs)=3

    parser.add_argument('--savefig', action='store_true')

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

    for prior_var in [1, 1e-1, 1e-2, 1e-4]:

        Hs_list = []
        method_list = []
        seed_list = []
        ev_list = []

        for i in range(len(args.Hs)):

            H = args.Hs[i]

            for seed in [1]:

                for method in ['nf_gamma','nf_gammatrunc','truth']:

                    if method == 'truth':

                        path = '{}/{}_{}_n5000_H{}_seed{}_prior{}_varparams{}'\
                            .format(args.path, 'nf_gamma', args.dataset, H, seed, prior_var, 'allones')
                        ev_list += [torch.load('{}/results.pt'.format(path))['asy_log_pDn']]
                        method_list += [method]

                    else:

                        for varparams in ['allones', 'icml', 'exp', 'allones']:

                            path = '{}/{}_{}_n5000_H{}_seed{}_prior{}_varparams{}'\
                                .format(args.path, method, args.dataset, H, seed, prior_var, varparams)
                            ev_list += [torch.load('{}/results.pt'.format(path))['elbo'].detach().numpy()
                                        + torch.load('{}/args.pt'.format(path))['nSn'].numpy()]
                            method_list += ['{}_{}'.format(method,varparams)]

                    seed_list += ['{}'.format(seed)]
                    Hs_list += [H]
                    #
                    #         # elbo_hist_nf_gammatrunc = torch.load('{}/results.pt'.format(path))['elbo_hist']
                    #         # elbo_hist_nf_gaussian = torch.load('{}/results.pt'.format(path))['elbo_hist']
                    #
                    #         # plt.plot(elbo_hist_nf_gaussian,'r',label='gaussian')
                    #         # plt.plot(elbo_hist_nf_gammatrunc,label='gammatrunc')
                    #         # plt.title('dataset {} H {} seed {}'.format(args.dataset, H, seed))
                    #         # plt.show()

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


        if args.savefig:
            # plt.savefig('{}/{}_n{}_prior{}.pgf'.format(args.path, args.dataset, n, prior_var), bbox_inches='tight')
            plt.savefig('{}/{}_prior{}.png'.format(args.path, args.dataset, prior_var), bbox_inches='tight')

        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    main()

