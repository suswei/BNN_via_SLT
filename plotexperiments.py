import seaborn as sns
import torch
from matplotlib import pyplot as plt
import argparse
from scipy import stats
import numpy as np
from dataset_factory import get_lmbda

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='?')
    # parser.add_argument('--dataset', default='reducedrank', type=str, choices=['reducedrank', 'tanh'])
    parser.add_argument('--savefig', action='store_true')
    parser.add_argument('--path', type=str)
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

    hyperparameter_experiments = torch.load('{}/hyp.pt'.format(args.path))
    tasks = hyperparameter_experiments.__len__()

    Hs_list = []
    method_list = []
    seed_list = []
    ev_list = []
    ns_list = []
    priorvar_list = []
    dataset_list = []
    layers_list = []

    for taskid in range(tasks):

        path = '{}/taskid{}/'.format(args.path, taskid)
        try:

            results = torch.load('{}/results.pt'.format(path),  map_location=torch.device('cpu'))
            sim_args = torch.load('{}/args.pt'.format(path), map_location=torch.device('cpu'))

            ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
            ns_list += [sim_args['sample_size']]

            # method_list += ['{}_k{}_l{}_{}'.format(sim_args['method'], sim_args['k0'], sim_args['lmbda0'], sim_args['no_couplingpairs'])]
            if sim_args['method'] == 'nf_gaussian':
                method_list += ['{}_{}'.format(sim_args['method'], sim_args['no_couplingpairs'])]
            elif sim_args['method'] == 'nf_gamma':
                method_list += ['{}_pairs{}'.format(sim_args['method'], sim_args['no_couplingpairs'])]
            seed_list += [sim_args['seed']]
            Hs_list += [sim_args['H']]
            priorvar_list += [sim_args['prior_var']]
            dataset_list += [sim_args['dataset']]
            layers_list += [sim_args['no_couplingpairs']]
        except:
            print('missing taskid {}'.format(taskid))

    summary_pd = pd.DataFrame({'dataset': dataset_list,
                               'H': Hs_list,
                               'ELBOplusnSn': ev_list,
                               'n': ns_list,
                               'method': method_list,
                               'prior_var': priorvar_list,
                               'no_couplingpairs': layers_list,
                               'seed': seed_list})

    unique_ns = list(set(summary_pd['n']))
    unique_datasets = list(set(summary_pd['dataset']))

    for dataset in unique_datasets:

        temp = summary_pd.loc[summary_pd['dataset'] == dataset]
        unique_Hs = list(set(temp['H']))

        for n in unique_ns:

            for H in unique_Hs:

                for taskid in range(tasks):

                    try:
                        path = '{}/taskid{}/'.format(args.path, taskid)
                        results = torch.load('{}/results.pt'.format(path), map_location=torch.device('cpu'))
                        sim_args = torch.load('{}/args.pt'.format(path), map_location=torch.device('cpu'))

                        if sim_args['H'] == H and sim_args['sample_size']==n:
                            ev_list += [results['asy_log_pDn']]
                            method_list += ['truth']
                            seed_list += [sim_args['seed']]
                            Hs_list += [H]
                            ns_list += [sim_args['sample_size']]
                            priorvar_list += [sim_args['prior_var']]
                            dataset_list += [sim_args['dataset']]
                            layers_list += [sim_args['no_couplingpairs']]

                    except:
                        print('missing taskid {}'.format(taskid))

    # method_list = pd.Series(method_list, dtype='category')
    # seed_list = pd.Series(seed_list, dtype="category")

    summary_pd = pd.DataFrame({'dataset': dataset_list,
                               'H': Hs_list,
                               'ELBOplusnSn': ev_list,
                               'n': ns_list,
                               'method': method_list,
                               'prior_var': priorvar_list,
                               'no_couplingpairs': layers_list,
                               'seed': seed_list})

    # summary_pd = summary_pd.loc[summary_pd['n']!=13360]
    summary_pd = summary_pd.dropna()
    summary_pd = summary_pd.loc[summary_pd['ELBOplusnSn']>=-1e+4] # remove instances where convergence was clearly not reached

    unique_methods = list(set(summary_pd['method']))
    unique_layers = list(set(summary_pd['no_couplingpairs']))

    # prior hyperparamater versus ELBO + nSn
    for H in list(set(summary_pd['H'])):

        for n in list(set(summary_pd['n'])):

            temp = summary_pd.loc[(summary_pd['H'] == H) & (summary_pd['n'] == n)]
            print('H={} n={}'.format(H, n))
            print(temp.groupby(['prior_var', 'method'])['ELBOplusnSn'].mean())
            print(temp.groupby(['prior_var', 'method'])['ELBOplusnSn'].std())
            std = temp.groupby(['prior_var', 'method'])['ELBOplusnSn'].std().unstack()
            try:
                temp.groupby(['prior_var','method'])['ELBOplusnSn'].mean().unstack().plot(yerr=std)
                # temp.groupby(['prior_var', 'method'])['ELBOplusnSn'].mean().unstack().plot()

                plt.title('H = {} n ={}'.format(H, n))
                plt.savefig('{}/tanh{}_n{}.png'.format(args.path, H, n))
                # plt.show()
                plt.close()
            except:
                print('blah')

    # log n slope plot
    for dataset in unique_datasets:

        temp = summary_pd.loc[summary_pd['dataset'] == dataset]
        unique_priorvars = list(set(temp['prior_var']))
        unique_Hs = list(set(temp['H']))

        for prior_var in unique_priorvars:


            temp2 = temp.loc[(temp['prior_var'] == prior_var) & (temp['dataset'] == dataset)]
                # for n in unique_ns:
                #     current_pd = temp2.loc[temp2['n'] == n]
                #
                #     # barplot - hold dataset, n, no_couplingpairs, prior_var
                #     g = sns.barplot(x="H", y="ELBOplusnSn",
                #                     hue="method",
                #                     data=current_pd)
                #
                #     sns.set_style("ticks")
                #     hatches = ['/', '/', '/',
                #                '+', '+', '+',
                #                'x', 'x', 'x']
                #     for hatch, patch in zip(hatches, g.patches):
                #         patch.set_hatch(hatch)
                #     leg = plt.legend(bbox_to_anchor=(1, 1), loc=2)
                #     for patch in leg.get_patches():
                #         patch.set_height(12)
                #         patch.set_y(-6)
                #
                #     plt.title(
                #         '{} n {} prior_var {} no_couplinglayers {} nett_tanh {}'
                #         .format(dataset, n, prior_var, no_couplingpairs, nett_tanh))
                #
                #     if args.savefig:
                #         plt.savefig(
                #             '{}/barplot_{}n{}layer{}nett{}prior{}.png'.format(args.path, dataset, n, no_couplingpairs, nett_tanh,
                #                                                   prior_var), bbox_inches='tight')
                #     plt.show()
                #     plt.close()

            for H in unique_Hs:

                truth = get_lmbda([H], dataset)[0]

                for method in unique_methods:

                    current_pd = temp2.loc[(temp2['H'] == H) & (temp2['method']==method)]
                    print(current_pd)

                    # print(current_pd[(np.abs(stats.zscore(current_pd['ELBOplusnSn'])) < 1)])

                    evs = current_pd.groupby('n')['ELBOplusnSn'].mean()
                    print(evs)

                    if len(evs._index)>1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(evs._index), evs.values)
                        plt.plot(np.log(evs._index), evs.values, '.')
                        plt.title('{} H {}: method {} prior_var {}, \n truth {} versus slope {:2f}, intercept {:2f} and R2 {:2f}'
                                  .format(dataset, H, method, prior_var, -truth, slope, intercept, r_value))

                        if args.savefig:
                            plt.savefig('{}/{}{}_{}_prior{:.4f}.png'.format(args.path, dataset, H, method, prior_var), bbox_inches='tight')
                        # plt.show()
                        plt.close()


if __name__ == "__main__":
    main()

