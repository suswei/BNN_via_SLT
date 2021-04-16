import seaborn as sns
import torch
import pandas as pd
from matplotlib import pyplot as plt
import argparse


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--dataset_of_interest', type=str, choices=['reducedrank', 'tanh'])

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

    Hs_list = []
    method_list = []
    seed_list = []
    ev_list = []

    prior_of_interest = 1e-1
    dataset_of_interest = args.dataset_of_interest

    for taskid in range(269):

        path = 'taskid{}/'.format(taskid)
        try:
            results = torch.load('{}/results.pt'.format(path))
            sim_args = torch.load('{}/args.pt'.format(path))

            if prior_of_interest == 'unif':

                if sim_args['prior'] == prior_of_interest and sim_args['dataset'] == dataset_of_interest:

                    ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
                    method_list += [sim_args['method']]
                    seed_list += [sim_args['seed']]
                    Hs_list += [sim_args['H']]
            else:

                if sim_args['prior_var'] == prior_of_interest and sim_args['dataset'] == dataset_of_interest:

                    ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
                    method_list += [sim_args['method']]
                    seed_list += [sim_args['seed']]
                    Hs_list += [sim_args['H']]
        except:
            break

    method_list = pd.Series(method_list, dtype='category')
    seed_list = pd.Series(seed_list, dtype="category")

    summary_pd = pd.DataFrame({'H': Hs_list,
                               'ELBOplusnSn': ev_list,
                               'method': method_list,
                               'seed': seed_list})

    g = sns.barplot(x="H", y="ELBOplusnSn",
                    hue="method",
                    data=summary_pd)

    sns.set_style("ticks")
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
        if prior_of_interest == 'unif':
            plt.savefig('{}_unifprior.png'.format(args.dataset), bbox_inches='tight')
        else:
            plt.savefig('{}_priorvar{}.png'.format(args.dataset, args.prior_var), bbox_inches='tight')
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()

