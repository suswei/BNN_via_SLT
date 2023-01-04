import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [int(round(np.exp(4))) * 32, int(round(np.exp(4.5))) * 32, int(round(np.exp(5))) * 32, int(round(np.exp(5.5))) * 32, int(round(np.exp(5.75))) * 32, int(round(np.exp(6.0))) * 32]

    tanh_Hs = [15, 115, 280, 1280, 3690]
    rr_Hs = [2, 16, 40, 180, 525]
    ffrelu_Hs = [3, 10, 16, 35, 60]

    nfs = ['2 4', '2 16', '4 4', '4 16']
    lrs = [0.001, 0.01]

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh_zeromean', 'tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':  lrs,
        'nf': nfs
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
        'lr':   lrs,
        'nf': nfs
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
        'lr':  lrs,
        'nf': nfs
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments, tanh_Hs, rr_Hs


def main(taskid):

    hyperparameter_experiments, _, _ = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'results/{}/taskid{}/'.format(temp['dataset'], taskid)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(hyperparameter_experiments, 'results/{}/hyp.pt'.format(temp['dataset']))

    os.system("python3 main.py "
              "--data %s %s %s "
              "--var_mode %s %s %s "
              "--trainR 5 "
              "--lr %s "
              "--epochs 2000 " 
              "--display_interval 1000 "
              "--seeds 1 2 3 4 5 6 7 8 9 10"
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'],
                 temp['base_dist'], temp['nf'], temp['grad_flag'], temp['lr'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
