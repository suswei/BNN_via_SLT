import seaborn as sns
import torch
from matplotlib import pyplot as plt
import argparse

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# run separately for different datasets
def main():

    parser = argparse.ArgumentParser(description='?')
    parser.add_argument('--savefig', action='store_true')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    hyperparameter_experiments = torch.load('{}/hyp.pt'.format(args.path))
    tasks = hyperparameter_experiments.__len__()

    dataset_list = []
    Hs_list = []
    ns_list = []
    zeromean_list = []
    seed_list = []
    priorvar_list = []
    method_short_list = []
    method_list = []
    ev_list = []

    ####################################################################################################################
    for taskid in range(tasks):

        path = '{}/taskid{}/'.format(args.path, taskid)
        try:

            results = torch.load('{}/results.pt'.format(path),  map_location=torch.device('cpu'))
            sim_args = torch.load('{}/args.pt'.format(path), map_location=torch.device('cpu'))

            dataset_list += [sim_args['dataset']]
            Hs_list += [sim_args['H']]
            ns_list += [sim_args['sample_size']]
            zeromean_list += [sim_args['zeromean']]
            seed_list += [sim_args['seed']]
            priorvar_list += [sim_args['prior_var']]

            ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
            method_list += ['{}_{}_{}_{}_{}'.format(sim_args['method'], sim_args['nf_couplingpair'], sim_args['nf_hidden'],
                                                         sim_args['var_mode'][3], sim_args['var_mode'][4])] # TODO: change to var_var_mode
            if sim_args['method'] == 'nf_gamma':
                method_short_list += ['gamma']
            else:
                method_short_list += ['gaussian']

        except:
            print('missing taskid {}'.format(taskid))

    for taskid in range(tasks):

        path = '{}/taskid{}/'.format(args.path, taskid)
        try:

            results = torch.load('{}/results.pt'.format(path),  map_location=torch.device('cpu'))
            sim_args = torch.load('{}/args.pt'.format(path), map_location=torch.device('cpu'))

            dataset_list += [sim_args['dataset']]
            Hs_list += [sim_args['H']]
            ns_list += [sim_args['sample_size']]
            zeromean_list += [sim_args['zeromean']]
            seed_list += [sim_args['seed']]
            priorvar_list += [sim_args['prior_var']]

            ev_list += [results['asy_log_pDn']]
            method_short_list += ['truth']
            method_list += ['truth']

        except:
            print('missing taskid {}'.format(taskid))

    summary_pd = pd.DataFrame({'dataset': dataset_list,
                               '$H$': Hs_list,
                               'n': ns_list,
                               'zeromean': zeromean_list,
                               'seed': seed_list,
                               'prior_var': priorvar_list,
                               'method_short': method_short_list,
                               'method': method_list,
                               '$\Psi(q^*,g^*)$': ev_list,
                               })
    ####################################################################################################################

    summary_pd = summary_pd.drop_duplicates()
    summary_pd = summary_pd.dropna()
    summary_pd = summary_pd.loc[summary_pd['$\Psi(q^*,g^*)$']>=-1e+6] # remove instances where convergence was clearly not reached

    ####################################################################################################################

    unique_ns = list(set(ns_list))
    unique_zeromean = list(set(zeromean_list))
    unique_priorvars = list(set(priorvar_list))

    for n, prior_var, zeromean in [(n, prior_var, zeromean)
                                      for n in unique_ns
                                      for prior_var in unique_priorvars
                                      for zeromean in unique_zeromean]:

        temp = summary_pd.loc[(summary_pd['n'] == n) & (summary_pd['prior_var'] == prior_var) & (summary_pd['zeromean'] == zeromean)]

        title = '{}_n{}_zeromean{}_priorvar{}'.format(args.path, n, zeromean, prior_var)
        sns.catplot(x="$H$", y="$\Psi(q^*,g^*)$", hue="method", kind="bar", data=temp,
                    size=6, palette="muted",
                    legend_out=False,
                    )
        plt.savefig('output/{}_catplot.png'.format(title))


if __name__ == "__main__":
    main()

