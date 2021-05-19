import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []
    sample_sizes = (np.round(np.exp([8.5,8.75,9.0,9.25]))).astype(int)
    seeds = [1, 2, 3, 4, 5]
    no_couplingpairs = [10]
    varparams_modes = ['a0', 'allones']

    tanh_Hs = [1600, 6400]
    rr_Hs = [40, 80]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gamma'],
        'no_couplingpairs': no_couplingpairs,
        'nf_gamma_mode': varparams_modes,
        'prior_var': [1e-2],
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
        'nf_gamma_mode': ['na'],
        'prior_var': [1e-2],
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gamma'],
        'no_couplingpairs': no_couplingpairs,
        'nf_gamma_mode': varparams_modes,
        'prior_var': [1, 1e-1],
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gaussian'],
        'no_couplingpairs': no_couplingpairs,
        'nf_gamma_mode': ['na'],
        'prior_var': [1, 1e-1],
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'comp_nett_tanh'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--no_couplingpairs %s  --nf_gamma_mode %s --nett_tanh true "
              " --exact_EqLogq --epochs 3000 --trainR 5 --display_interval 10 "
              "--dataset %s --sample_size %s --zeromean True "
              "--method %s "
              "--beta_star "
              "--H %s "
              "--prior_var %s "
              "--seed %s "
              "--path %s "
              % (temp['no_couplingpairs'], temp['nf_gamma_mode'],
                 temp['dataset'], temp['sample_size'],
                 temp['method'],
                 temp['H'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
