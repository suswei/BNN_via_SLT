import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []

    dataset = ['tanh']
    Hs = [16, 64, 256, 1024]
    sample_sizes = [5000]
    zeromeans = ['True','False']
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    prior_vars = [1, 100]

    nf_couplingpairs = [2]
    no_hiddens = [16]

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_var': prior_vars,
        'method': ['nf_gaussian'],
        'varparam0': ['0 1', '1 .001'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_var': prior_vars,
        'method': ['nf_gamma'],
        'varparam0': ['1000 1'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
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
    #     'nf_couplingpair': nf_couplingpair,
    #     'seed': seeds,
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'H': rr_Hs,
    #     'sample_size': sample_sizes,
    #     'method': ['nf_gamma'],
    #     'nf_couplingpair': nf_couplingpair,
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
    #     'nf_couplingpair': nf_couplingpair,
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

    path = 'tanh_2d45484'
    # if not os.path.exists(path):
        # os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--data %s %s %s %s "
              "--mode %s %s %s %s "
              "--epochs 500 --display_interval 100 "
              "--prior_dist gaussian %s "
              "--seed %s "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'], temp['zeromean'],
                 temp['method'], temp['nf_couplingpair'], temp['nf_hidden'], temp['varparam0'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
