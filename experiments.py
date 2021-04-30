import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []
    methods = ['nf_gamma']
    modes = ['icml', 'allones']
    sample_sizes = (np.round(np.exp([7.0, 7.5, 8.0, 8.5]))).astype(int)
    seeds = [1, 2, 3]

    tanh_Hs = [4, 9, 16]
    rr_Hs = [40, 80]

    ############################################  GAUSSIAN PRIOR -- NF_GAMMA ########################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'sample_size': sample_sizes,
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior': ['gaussian'],
        'seed': seeds,
        'K0net': ['True','False']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': methods,
    #     'nf_gamma_mode': modes,
    #     'H': rr_Hs,
    #     'prior': ['gaussian'],
    #     'prior_var': prior_vars,
    #     'seed': seeds
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ################################################# UNIF PRIOR -- NF_GAMMA ###################################################

    # hyperparameter_config = {
    #     'dataset': ['tanh'],
    #     'sample_size': sample_sizes,
    #     'method': methods,
    #     'nf_gamma_mode': modes,
    #     'H': tanh_Hs,
    #     'prior': ['unif'],
    #     'prior_var': [0],
    #     'seed': seeds,
    #     'zeromean': ['True', 'False']
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    #
    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': methods,
    #     'nf_gamma_mode': modes,
    #     'H': rr_Hs,
    #     'prior': ['unif'],
    #     'prior_var': [0],
    #     'seed': seeds
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
        'prior': ['gaussian'],
        'seed': seeds,
        'K0net': ['True', 'False']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': ['nf_gaussian'],
    #     'nf_gamma_mode': ['icml'],
    #     'H': rr_Hs,
    #     'prior': ['gaussian'],
    #     'prior_var': prior_vars,
    #     'seed': seeds
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ################################################# UNIF PRIOR -- NF_GAUSSIAN ###################################################

    # hyperparameter_config = {
    #     'dataset': ['tanh'],
    #     'method': ['nf_gaussian'],
    #     'nf_gamma_mode': ['icml'],
    #     'H': tanh_Hs,
    #     'prior': ['unif'],
    #     'prior_var': [0],
    #     'seed': seeds,
    #     'zeromean': ['True', 'False']
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    #
    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': ['nf_gaussian'],
    #     'nf_gamma_mode': ['icml'],
    #     'H': rr_Hs,
    #     'prior': ['unif'],
    #     'prior_var': [0],
    #     'seed': seeds
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'lowH_lognslope_coupling'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--nf vanilla_rnvp --nf_layers 5 --beta_star --exact_EqLogq --epochs 1000 --trainR 1 "
              "--dataset %s --sample_size %s --zeromean True "
              "--method %s "
              "--nf_gamma_mode %s "
              "--K0net %s "
              "--H %s "
              "--prior %s "
              "--seed %s "
              "--path %s "
              % (temp['dataset'], temp['sample_size'],
                 temp['method'],
                 temp['nf_gamma_mode'], temp['K0net'],
                 temp['H'],
                 temp['prior'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
