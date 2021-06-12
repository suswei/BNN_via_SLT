import sys
import os
import itertools
import torch
import numpy as np
from dataset_factory import get_lmbda


def set_sweep_config():

    hyperparameter_experiments = []

    tanh_Hs = [6400]
    sample_sizes = (np.round(np.exp([7.0, 7.5, 8.0, 8.5]))).astype(int)
    seeds = [1, 2, 3, 4, 5]

    no_couplingpairs = [10]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gammatrunc'],
        'no_couplingpairs': no_couplingpairs,
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gaussian'],
        'no_couplingpairs': no_couplingpairs,
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'H': rr_Hs,
    #     'sample_size': sample_sizes,
    #     'method': ['nf_gamma'],
    #     'no_couplingpairs': no_couplingpairs,
    #     'nf_gamma_mode': varparams_modes,
    #     'prior_var': [1e-1, 1e-2],
    #     'seed': seeds,
    #     'nett_tanh': ['true', 'false']
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'H': rr_Hs,
    #     'sample_size': sample_sizes,
    #     'method': ['nf_gaussian'],
    #     'no_couplingpairs': no_couplingpairs,
    #     'nf_gamma_mode': ['na'],
    #     'prior_var': [1e-1, 1e-2],
    #     'seed': seeds,
    #     'nett_tanh': ['true', 'false']
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'tanh6400'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    if temp['method'] == 'nf_gaussian':
        os.system("python3 main.py "
                  "--mode %s %s 128 0 1 "
                  "--exact_EqLogq --epochs 1000 --display_interval 100 "
                  "--data %s %s %s True "
                  "--prior_dist gaussian 1e-4 "
                  "--seed %s "
                  "--path %s "
                  % (temp['method'], temp['no_couplingpairs'],
                     temp['dataset'], temp['H'], temp['sample_size'],
                     temp['seed'],
                     path))

    elif temp['method'] == 'nf_gamma':

        os.system("python3 main.py "
                  "--mode %s %s 128 "
                  "--exact_EqLogq --epochs 1000 --display_interval 100 --trainR 10 "
                  "--data %s %s %s True "
                  "--prior_dist gaussian %s "
                  "--seed %s "
                  "--path %s "
                  % (temp['method'], temp['no_couplingpairs'],
                     temp['dataset'], temp['H'], temp['sample_size'],
                     temp['prior_var'],
                     temp['seed'],
                     path))


if __name__ == "__main__":
    main(sys.argv[1:])
