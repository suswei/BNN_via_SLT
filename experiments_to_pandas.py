import torch
import argparse
import os
import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# pd.options.display.float_format = '{:.2f}'.format
pd.set_option("display.precision", 4)

import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser(description='puts experimental results into a pandas dataframe')
    parser.add_argument('--path', type=str, help='folder that contains hyp.pt, taskid folders, and seeds within each taskid')
    args = parser.parse_args()

    hyperparameter_experiments = torch.load('{}/hyp.pt'.format(args.path))
    tasks = hyperparameter_experiments.__len__()

    dataset_list = []
    Hs_list = []
    wdim_list = []
    qdim_list = []
    trueRLCT_list = []
    ns_list = []
    seed_list = []
    prior_list = []
    method_list = []
    gradflag_list = []
    lr_list = []

    asy_list = []
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
                    ns_list += [np.log(sim_args['sample_size'])]
                    seed_list += [sim_args['seed']]

                    Hs_list += [sim_args['H']]
                    wdim_list += [sim_args['w_dim']]
                    qdim_list += [sim_args['qdim']]

                    trueRLCT_list += [sim_args['trueRLCT']]

                    prior_list += ['({}, {})'.format(sim_args['prior_mean'],sim_args['prior_var'])]
                    method_list += ['{}\_{}\_{}'.format(sim_args['base_dist'], sim_args['nf_couplingpair'], sim_args['nf_hidden'])]
                    gradflag_list += [sim_args['grad_flag']]
                    lr_list += [sim_args['lr']]

                    # evaluation metrics
                    ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
                    predloglik_list += [results['test_lpd'].detach().numpy()]
                    asy_list += [results['asy_log_pDn']]

                except:
                    print('missing taskid {}'.format(taskid))

    df = pd.DataFrame({'dataset': dataset_list,
                               '$H$': Hs_list, '$dim_w$': wdim_list, '$\lambda$': trueRLCT_list,
                               '$\log n$': ns_list,
                               'seed': seed_list,
                               'method': method_list,  '$dim_q$': qdim_list,
                               'lr': lr_list, 'grad_flag': gradflag_list,
                               '$-\lambda \log n$': asy_list,
                               'ELBO+$nS_n$': ev_list,
                               'test_lpd': predloglik_list,
                               })

    df['test_lpd'] = df['test_lpd'].astype(float)

    df.to_pickle("summary_{}.pkl".format(args.path))
    # df = pd.read_pickle("my_data.pkl")

    ####################################################################################################################

    # unique_ns = list(set(ns_list))
    # 
    # for n in unique_ns:
    # 
    #     temp = df.loc[(df['$\log n$'] == n)]
    #     title = '{}_n{}'.format(args.path, n)
    # 
    #     print(title)
    #     pdsave = temp.groupby(['$H$', 'method'])['ELBO+$nS_n$'].describe()
    #     print(pdsave)
    #     with open('output/{}.tex'.format(title), 'w') as tf:
    #         tf.write(pdsave.to_latex(escape=False,  float_format="%.2f", columns=['count', 'mean', 'std'], multirow=True))

    # if not os.path.exists('output'):
    #     os.makedirs('output')
    #
    for metric in ['ELBO+$nS_n$', 'test_lpd']:
        pdsave = df.groupby(['$\log n$', '$H$', 'method','lr'])[metric].describe()
        print(pdsave)
    #     with open('output/{}_{}.tex'.format(args.path, metric), 'w') as tf:
    #         tf.write(pdsave.to_latex(escape=False,  float_format="%.2f", columns=['count', 'mean', 'std'], multirow=True))

    unique_Hs = list(set(Hs_list))

    for H in unique_Hs:
        temp = df.loc[(df['$H$'] == H)]
        temp.set_index('$\log n$', inplace=True)

        for metric in ['test_lpd', 'ELBO+$nS_n$']:
            # fig, ax = plt.subplots()
            temp.groupby('method')[metric].plot(legend=True, style='o-')
            plt.title('{} H={}'.format(args.path, H))
            plt.ylabel('{}'.format(metric))
            plt.show()
            # for key, group in temp.groupby('method'):
            #     group.plot('$\log n$', metric, label=key, ax=ax)
            # plt.show()


if __name__ == "__main__":
    main()

