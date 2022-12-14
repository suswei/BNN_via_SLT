import torch
import argparse
import os

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format


# run separately for different datasets
def main():

    parser = argparse.ArgumentParser(description='puts experimental results into a pandas data frame')
    parser.add_argument('--path', type=str, help='tanh or reducedrank')
    args = parser.parse_args()

    hyperparameter_experiments = torch.load('{}/hyp.pt'.format(args.path))
    tasks = hyperparameter_experiments.__len__()

    dataset_list = []
    Hs_list = []
    ns_list = []
    seed_list = []
    prior_list = []
    method_list = []
    ev_list = []
    predloglik_list = []

    ####################################################################################################################
    for taskid in range(tasks):

        path = '{}/taskid{}/'.format(args.path, taskid)
        for root, subdirectories, files in os.walk(path):
            for subdirectory in subdirectories:
                current_path = os.path.join(root, subdirectory)
                try:

                    results = torch.load('{}/results.pt'.format(current_path),  map_location=torch.device('cpu'))
                    sim_args = torch.load('{}/args.pt'.format(current_path), map_location=torch.device('cpu'))

                    dataset_list += [sim_args['dataset']]
                    Hs_list += [sim_args['H']]
                    ns_list += [sim_args['sample_size']]
                    seed_list += [sim_args['seed']]
                    prior_list += ['({}, {})'.format(sim_args['prior_mean'],sim_args['prior_var'])]
                    method_list += ['{}\_{}'.format(sim_args['base_dist'], sim_args['grad_flag'])]

                    # evaluation metrics
                    ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
                    predloglik_list += [results['predloglik'].detach().numpy()]

                except:
                    print('missing taskid {}'.format(taskid))

    ####################################################################################################################

    # for taskid in range(tasks):
    #
    #     path = '{}/taskid{}/'.format(args.path, taskid)
    #     for root, subdirectories, files in os.walk(path):
    #         for subdirectory in subdirectories:
    #             current_path = os.path.join(root, subdirectory)
    #             try:
    #
    #                 results = torch.load('{}/results.pt'.format(current_path),  map_location=torch.device('cpu'))
    #                 sim_args = torch.load('{}/args.pt'.format(current_path), map_location=torch.device('cpu'))
    #
    #                 dataset_list += [sim_args['dataset']]
    #                 Hs_list += [sim_args['H']]
    #                 ns_list += [sim_args['sample_size']]
    #                 seed_list += [sim_args['seed']]
    #                 prior_list += ['({}, {})'.format(sim_args['prior_mean'], sim_args['prior_var'])]
    #
    #                 method_list += ['$-\lambda \log n + (m-1) \log \log n$']
    #
    #                 ev_list += [results['asy_log_pDn']]
    #                 predloglik_list += [results['predloglik'].detach().numpy()]
    #
    #
    #             except:
    #                 print('missing taskid {}'.format(taskid))

    ####################################################################################################################

    summary_pd = pd.DataFrame({'dataset': dataset_list,
                               '$H$': Hs_list,
                               '$n$': ns_list,
                               'seed': seed_list,
                               'method': method_list,
                               'elbo+$nS_n$': ev_list,
                               'predloglik': predloglik_list
                               })

    # summary_pd = summary_pd.drop_duplicates()
    summary_pd = summary_pd.dropna()
    summary_pd = summary_pd.loc[summary_pd['elbo+$nS_n$']>=-1e+6] # remove instances where convergence was clearly not reached

    ####################################################################################################################

    # unique_ns = list(set(ns_list))
    # 
    # for n in unique_ns:
    # 
    #     temp = summary_pd.loc[(summary_pd['$n$'] == n)]
    #     title = '{}_n{}'.format(args.path, n)
    # 
    #     print(title)
    #     pdsave = temp.groupby(['$H$', 'method'])['elbo+$nS_n$'].describe()
    #     print(pdsave)
    #     with open('output/{}.tex'.format(title), 'w') as tf:
    #         tf.write(pdsave.to_latex(escape=False,  float_format="%.2f", columns=['count', 'mean', 'std'], multirow=True))

    if not os.path.exists('output'):
        os.makedirs('output')

    for metric in ['elbo+$nS_n$', 'predloglik']:
        pdsave = summary_pd.groupby(['$n$', '$H$', 'method'])[metric].describe()
        print(pdsave)
        with open('output/{}_{}.tex'.format(args.path, metric), 'w') as tf:
            tf.write(pdsave.to_latex(escape=False,  float_format="%.2f", columns=['count', 'mean', 'std'], multirow=True))


if __name__ == "__main__":
    main()

