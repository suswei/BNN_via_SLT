import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []
    sample_sizes = (np.round(np.exp([8.75, 9.0, 9.25]))).astype(int)
    seeds = [1, 2, 3, 4, 5]
    no_couplingpairs = [10]
    lmbda0s = [1000]
    varparams_modes = ['pgamma']

    tanh_Hs = [1600, 900]
    rr_Hs = [40]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gamma'],
        'no_couplingpairs': no_couplingpairs,
        'nf_gamma_mode': varparams_modes,
        'prior_var': [1e-2],
        'lmbda0': lmbda0s,
        'seed': seeds,
        'nett_tanh': ['false']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    # hyperparameter_config = {        'seed': seeds,
    #     'dataset': ['tanh'],
    #     'H': tanh_Hs,
    #     'sample_size': sample_sizes,
    #     'method': ['nf_gaussian'],
    #     'no_couplingpairs': no_couplingpairs,
    #     'nf_gamma_mode': ['na'],
    #     'prior_var': [1e-2],
    #     'seed': seeds,
    #     'lmbda0': lmbda0s,
    #     'nett_tanh': ['true']
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

    path = 'mixed'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--no_couplingpairs %s  --nf_gamma_mode %s --nett_tanh %s --lmbda0 %s "
              " --exact_EqLogq --epochs 2000 --trainR 5 --display_interval 100 "
              "--dataset %s --sample_size %s --zeromean True "
              "--method %s "
              "--beta_star --lmbda_star "
              "--H %s "
              "--prior_var %s "
              "--seed %s "
              "--path %s "
              % (temp['no_couplingpairs'], temp['nf_gamma_mode'], temp['nett_tanh'], temp['lmbda0'],
                 temp['dataset'], temp['sample_size'],
                 temp['method'],
                 temp['H'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
