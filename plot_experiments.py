import seaborn as sns
import torch
from matplotlib import pyplot as plt
import argparse

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format

# run separately for different datasets
def main():

    parser = argparse.ArgumentParser(description='?')
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
            if sim_args['method'] == 'nf_gamma':
                method_str = 'nf\_gamma'
            elif sim_args['method'] == 'nf_gaussian':
                method_str = 'nf\_gaussian'

            method_list += ['{}\_{}\_{}\_{}\_{}'.format(method_str, sim_args['nf_couplingpair'], sim_args['nf_hidden'],
                                                         sim_args['var_mode'][3], sim_args['var_mode'][4])]
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

            if sim_args['dataset'] == 'reducedrank' or sim_args['zeromean'] == 'True':
                dataset_list += [sim_args['dataset']]
                Hs_list += [sim_args['H']]
                ns_list += [sim_args['sample_size']]
                zeromean_list += [sim_args['zeromean']]
                seed_list += [sim_args['seed']]
                priorvar_list += [sim_args['prior_var']]

                ev_list += [results['asy_log_pDn']]
                method_short_list += ['$-\lambda \log n + (m-1) \log \log n$']
                method_list += ['$-\lambda \log n + (m-1) \log \log n$']

        except:
            print('missing taskid {}'.format(taskid))

    summary_pd = pd.DataFrame({'dataset': dataset_list,
                               '$H$': Hs_list,
                               'n': ns_list,
                               'zeromean': zeromean_list,
                               'seed': seed_list,
                               r'$\sigma^2(\varphi)$': priorvar_list,
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

    # for n, prior_var, zeromean in [(n, prior_var, zeromean)
    #                                   for n in unique_ns
    #                                   for prior_var in unique_priorvars
    #                                   for zeromean in unique_zeromean]:

    for n, zeromean in [(n, zeromean)
                                   for n in unique_ns
                                   for zeromean in unique_zeromean]:

        # temp = summary_pd.loc[(summary_pd['n'] == n) & (summary_pd['prior_var'] == prior_var) & (summary_pd['zeromean'] == zeromean)]
        # title = '{}_n{}_zeromean{}_priorvar{}'.format(args.path, n, zeromean, prior_var)

        temp = summary_pd.loc[(summary_pd['n'] == n)  & (summary_pd['zeromean'] == zeromean)]
        title = '{}_n{}_zeromean{}'.format(args.path, n, zeromean)

        print(title)
        pdsave = temp.groupby(['$H$', r'$\sigma^2(\varphi)$', 'method'])['$\Psi(q^*,g^*)$'].describe()
        print(pdsave)
        with open('output/{}.tex'.format(title), 'w') as tf:
            tf.write(pdsave.to_latex(escape=False,  float_format="%.2f", columns=['count','mean','std'], multirow=True))


if __name__ == "__main__":
    main()

