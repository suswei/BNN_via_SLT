import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []
    sample_sizes = (np.round(np.exp([8.0, 8.25, 8.5]))).astype(int)
    no_couplingpairs = [2, 10]
    l0s = [1, 30, 100]

    tanh_Hs = [400]
    seeds = [1, 2, 3, 4, 5]

    rr_Hs = [40]

    prior_vars = np.arange(0.001, 0.03, 0.005).tolist()

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'prior_var': prior_vars,
        'method': ['nf_gamma'],
        'no_couplingpairs': no_couplingpairs,
        'lmbda0': l0s,
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'prior_var': prior_vars,
        'method': ['nf_gaussian'],
        'no_couplingpairs': no_couplingpairs,
        'nf_gamma_mode': ['na'],
        'seed': seeds,
        'lmbda0': [0],
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

    path = 'tanh400'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--no_couplingpairs %s --lmbda0 %s "
              " --exact_EqLogq --epochs 500 --display_interval 100 "
              "--dataset %s --sample_size %s "
              "--method %s "
              "--beta_star "
              "--H %s "
              "--prior_var %s "
              "--seed %s "
              "--path %s "
              % (temp['no_couplingpairs'], temp['lmbda0'],
                 temp['dataset'], temp['sample_size'],
                 temp['method'],
                 temp['H'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
