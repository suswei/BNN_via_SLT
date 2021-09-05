import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []

    tanh_Hs = [400]

    sample_sizes = (np.round(np.exp([8.5]))).astype(int)
    sample_sizes = [5000]

    seeds = [1, 2, 3, 4, 5]
    prior_vars = [1e-1]

    no_couplingpairs = [10]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'prior_var': prior_vars,
        'method': ['nf_gaussian'],
        'q_mean_var': ['0 1', '1 .1'],
        'no_couplingpairs': no_couplingpairs,
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'prior_var': prior_vars,
        'method': ['nf_gamma'],
        'q_mean': [0],
        'q_var': [.1],
        'no_couplingpairs': no_couplingpairs,
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    # hyperparameter_config = {
    #     'dataset': ['tanh'],
    #     'H': tanh_Hs,
    #     'sample_size': sample_sizes,
    #     'prior_var': prior_vars,
    #     'method': ['nf_gaussian'],
    #     'no_couplingpairs': no_couplingpairs,
    #     'seed': seeds,
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

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

    path = 'tanh'
    # if not os.path.exists(path):
        # os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--mode %s %s 128 %s "
              "--epochs 1000 --display_interval 10 "
              "--data %s %s %s True "
              "--prior_dist gaussian %s "
              "--seed %s "
              "--path %s "
              % (temp['method'], temp['no_couplingpairs'], temp['q_mean_var'],
                 temp['dataset'], temp['H'], temp['sample_size'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
