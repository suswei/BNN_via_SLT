import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []
    methods = ['nf_gamma']
    modes = ['allones']
    # sample_sizes = (np.round(np.exp([8.5, 8.6, 8.7, 8.8]))).astype(int)
    sample_sizes = (np.round(np.exp([7.5, 8.0, 8.5]))).astype(int)
    seeds = [1, 2, 3]

    tanh_Hs = [1600]
    rr_Hs = [40]

    ############################################  GAUSSIAN PRIOR -- NF_GAMMA ########################################################

    hyperparameter_config = {
        'dataset': ['tanh', 'tanh_general'],
        'sample_size': sample_sizes,
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior_var': [1e-2, 1e-3],
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'sample_size': sample_sizes,
        'method': methods,
        'nf_gamma_mode': modes,
        'H': rr_Hs,
        'prior_var': [1, 1e-1],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    #################################################  GAUSSIAN PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh', 'tanh_general'],
        'sample_size': sample_sizes,
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['icml'],
        'H': tanh_Hs,
        'prior_var': [1e-2, 1e-3],
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'sample_size': sample_sizes,
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['icml'],
        'H': rr_Hs,
        'prior_var': [1, 1e-1],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'neurips'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--nf rnvp --nf_layers 6  --exact_EqLogq --epochs 2000 --trainR 1 "
              "--dataset %s --sample_size %s --zeromean True "
              "--method %s "
              "--nf_gamma_mode %s --beta_star "
              "--H %s "
              "--prior_var %s "
              "--seed %s "
              "--path %s "
              % (temp['dataset'], temp['sample_size'],
                 temp['method'],
                 temp['nf_gamma_mode'],
                 temp['H'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
