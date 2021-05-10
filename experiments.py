import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []
    methods = ['nf_gamma']
    modes = ['allones']
    sample_sizes = (np.round(np.exp([8.5, 8.6, 8.7]))).astype(int)
    seeds = [1, 2, 3]
    layers = [6]

    tanh_Hs = [100]
    rr_Hs = [20]

    ############################################  GAUSSIAN PRIOR -- NF_GAMMA ########################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'sample_size': sample_sizes,
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'seed': seeds,
        'nf_layers': layers
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'sample_size': sample_sizes,
    #     'method': methods,
    #     'nf_gamma_mode': modes,
    #     'H': rr_Hs,
    #     'seed': seeds,
    #     'nf_layers': layers
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    #################################################  GAUSSIAN PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'sample_size': sample_sizes,
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['icml'],
        'H': tanh_Hs,
        'seed': seeds,
        'nf_layers': layers
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'sample_size': sample_sizes,
    #     'method': ['nf_gaussian'],
    #     'nf_gamma_mode': ['icml'],
    #     'H': rr_Hs,
    #     'seed': seeds,
    #     'nf_layers': layers
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'neurips'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--nf rnvp --nf_layers %s  --exact_EqLogq --epochs 2000 --trainR 5 --display_interval 10 "
              "--dataset %s --sample_size %s --zeromean True "
              "--method %s "
              "--nf_gamma_mode %s --beta_star "
              "--H %s "
              "--prior unif "
              "--seed %s "
              "--path %s "
              % (temp['nf_layers'],
                 temp['dataset'], temp['sample_size'],
                 temp['method'],
                 temp['nf_gamma_mode'],
                 temp['H'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
