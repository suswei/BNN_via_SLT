import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [int(round(np.exp(4))) * 32,
                    int(round(np.exp(4.25))) * 32,
                    int(round(np.exp(4.5))) * 32,
                    int(round(np.exp(4.75))) * 32,
                    int(round(np.exp(5.0))) * 32]
    nf_couplingpairs = [2, 4]
    no_hiddens = [4, 16]

    tanh_Hs = [99, 255]
    rr_Hs = [10, 16]
    ffrelu_Hs = [200, 512]

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh_zeromean', 'tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr': [0.01],
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
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':  [0.01],
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
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':  [0.01],
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
              "--trainR 10 "
              "--lr %s "
              "--epochs 2000 " 
              "--display_interval 1000 "
              "--seeds 1 2 3 4 5 "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'],
                 temp['base_dist'], temp['nf_couplingpair'], temp['nf_hidden'], temp['grad_flag'], temp['lr'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
