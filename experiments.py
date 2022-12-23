import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [int(round(np.exp(4))) * 32, int(round(np.exp(4.5))) * 32, int(round(np.exp(5))) * 32, int(round(np.exp(5.5))) * 32, int(round(np.exp(6.0))) * 32, int(round(np.exp(6.5))) * 32]
    nf_couplingpairs = [4, 8]
    no_hiddens = [16, 32]

    tanh_Hs = [15, 99, 255]
    rr_Hs = [4, 10, 16]
    ffrelu_Hs = [32, 200, 512]

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian_std'],
        'grad_flag': [False],
        'lr': [0.01, 0.001, 0.001],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian_std'],
        'grad_flag': [False],
        'lr':  [0.01, 0.001, 0.001],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    hyperparameter_config = {
        'dataset': ['ffrelu'],
        'H': ffrelu_Hs,
        'sample_size': sample_sizes,
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian_std'],
        'grad_flag': [False],
        'lr':  [0.01, 0.001, 0.001],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments, tanh_Hs, rr_Hs


def main(taskid):

    hyperparameter_experiments, _, _ = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = '{}/taskid{}/'.format(temp['dataset'], taskid)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(hyperparameter_experiments, '{}/hyp.pt'.format(temp['dataset']))

    os.system("python3 main.py "
              "--data %s %s %s "
              "--var_mode %s %s %s %s "
              "--lr %s "
              "--epochs 1500 " 
              "--display_interval 100 "
              "--seeds 1 2 3 4 5 6 7 8 9 10 "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'],
                 temp['base_dist'], temp['nf_couplingpair'], temp['nf_hidden'], temp['grad_flag'], temp['lr'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
