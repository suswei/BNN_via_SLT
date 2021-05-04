import seaborn as sns
import torch
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from scipy import stats
import numpy as np
from dataset_factory import get_lmbda


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--dataset_of_interest', default='reducedrank', type=str, choices=['reducedrank', 'tanh'])

    parser.add_argument('--savefig', action='store_true')

    parser.add_argument('--path_prefix', type=str)

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


    hyperparameter_experiments = torch.load('{}/hyp.pt'.format(args.path_prefix))
    tasks = hyperparameter_experiments.__len__()

    Hs_list = []
    method_list = []
    seed_list = []
    ev_list = []
    ns_list = []
    priorvar_list = []

    for taskid in range(tasks):

        path = '{}/taskid{}/'.format(args.path_prefix, taskid)
        try:
            results = torch.load('{}/results.pt'.format(path))
            sim_args = torch.load('{}/args.pt'.format(path))

            ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
            ns_list += [sim_args['sample_size']]

            if sim_args['method'] == 'nf_gaussian':
                method_list += [sim_args['method']]
            else:
                method_list += ['{}_{}'.format(sim_args['method'], sim_args['nf_gamma_mode'])]

            seed_list += [sim_args['seed']]
            Hs_list += [sim_args['H']]
            priorvar_list += [sim_args['prior_var']]

        except:
            print('missing taskid {}'.format(taskid))

    unique_Hs = list(set(Hs_list))
    unique_methods = list(set(method_list))
    unique_priorvars = list(set(priorvar_list))

    for H in unique_Hs:

        for taskid in range(tasks):

            try:
                path = '{}/taskid{}/'.format(args.path_prefix, taskid)
                results = torch.load('{}/results.pt'.format(path))
                sim_args = torch.load('{}/args.pt'.format(path))

                if sim_args['H'] == H:
                    ev_list += [results['asy_log_pDn']]
                    method_list += ['truth']
                    seed_list += [sim_args['seed']]
                    Hs_list += [H]
                    ns_list += [sim_args['sample_size']]
                    priorvar_list += [sim_args['prior_var']]

                    break

            except:
                print('missing taskid {}'.format(taskid))

    method_list = pd.Series(method_list, dtype='category')
    seed_list = pd.Series(seed_list, dtype="category")

    summary_pd = pd.DataFrame({'H': Hs_list,
                               'ELBOplusnSn': ev_list,
                               'n': ns_list,
                               'method': method_list,
                               'prior_var': priorvar_list,
                               'seed': seed_list})

    # log n slope plot
    for prior_var in unique_priorvars:

        for H in unique_Hs:

            for method in unique_methods:

                temp = summary_pd.loc[summary_pd['H'] == H]
                truth = get_lmbda([H], 'tanh')[0]
                temp = temp.loc[temp['method'] == method]
                temp = temp.loc[temp['prior_var'] == prior_var]

                evs = temp.groupby('n')['ELBOplusnSn'].mean()

                slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(evs._index), evs.values)
                plt.plot(np.log(evs._index), evs.values, '.')
                plt.title('H {}: method {} prior_var {}, \n truth {} versus slope {:2f} and R2 {:2f}'
                          .format(H, method, prior_var, -truth, slope, r_value))

                if args.savefig:
                    plt.savefig('{}/{}H{}prior{}'.format(args.path_prefix, method, H, prior_var*1000), bbox_inches='tight')

                plt.show()
                plt.close()

    # g = sns.barplot(x="H", y="ELBOplusnSn",
    #                 hue="method",
    #                 data=summary_pd)
    #
    # sns.set_style("ticks")
    # hatches = ['/', '/', '/',
    #            '+', '+', '+',
    #            'x', 'x', 'x']
    # for hatch, patch in zip(hatches, g.patches):
    #     patch.set_hatch(hatch)
    # leg = plt.legend(bbox_to_anchor=(1, 1), loc=2)
    # for patch in leg.get_patches():
    #     patch.set_height(12)
    #     patch.set_y(-6)
    #
    # if prior_of_interest == 'unif':
    #     title = '{}/{}_unifprior.png'.format(path_prefix, dataset_of_interest)
    # else:
    #     title = '{}/{}_priorvar{}.png'.format(path_prefix, dataset_of_interest, prior_of_interest)
    # plt.title(title)
    #
    # if args.savefig:
    #         plt.savefig(title, bbox_inches='tight')
    # else:
    #     plt.show()
    #
    # plt.close()


if __name__ == "__main__":
    main()

