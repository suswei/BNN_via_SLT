import torch
import argparse
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.precision", 4)


def main():

    parser = argparse.ArgumentParser(description='puts experimental results into a pandas dataframe')
    parser.add_argument('--path', type=str, help='folder that contains taskid folders, and seeds within each taskid')
    args = parser.parse_args()

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
    normalized_mvfe_list = []
    vge_list = []

    ####################################################################################################################
    for taskid in range(len(os.listdir('{}'.format(args.path)))):

        path = '{}/taskid{}/'.format(args.path, taskid)
        for root, subdirectories, files in os.walk(path):
            for subdirectory in subdirectories:
                current_path = os.path.join(root, subdirectory)

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
                normalized_mvfe_list += [- results['elbo'].detach().numpy() - sim_args['nSn'].numpy()]
                vge_list += [-sim_args['nSn_val'].numpy()/sim_args['val_size'] - results['test_lpd']]
                asy_list += [results['asy_log_pDn']]

    df = pd.DataFrame({'dataset': dataset_list,
                               '$H$': Hs_list, '$dim_w$': wdim_list, '$\lambda$': trueRLCT_list,
                               '$\log n$': ns_list,
                               'seed': seed_list,
                               'method': method_list,  '$dim_q$': qdim_list,
                               'lr': lr_list,
                               'grad_flag': gradflag_list,
                               '$-\lambda \log n$': asy_list,
                               'normalized MVFE': normalized_mvfe_list,
                               'VGE': vge_list,
                               })

    df['VGE'] = df['VGE'].astype(float)
    df.to_pickle("{}/{}.pkl".format(args.path, args.path))


if __name__ == "__main__":
    main()

