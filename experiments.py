import sys
import os
import itertools
import torch
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []
    sample_sizes = (np.round(np.exp([8.5, 9.0, 9.5, 10.0, 10.5]))).astype(int)
    seeds = [1, 2, 3, 4, 5]
    no_couplingpairs = [2, 5, 10]

    tanh_Hs = [400, 1600]
    rr_Hs = [20, 40]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gamma', 'nf_gaussian'],
        'no_couplingpairs': no_couplingpairs,
        'prior_var': [1e-1, 1e-2, 1e-4],
        'seed': seeds,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'method': ['nf_gamma', 'nf_gaussian'],
        'no_couplingpairs': no_couplingpairs,
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

    path = 'neurips'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--no_couplingpairs %s  --exact_EqLogq --epochs 2000 --trainR 1 --display_interval 100 "
              "--dataset %s --sample_size %s --zeromean True "
              "--method %s "
              "--beta_star "
              "--H %s "
              "--prior_var %s "
              "--seed %s "
              "--path %s "
              % (temp['no_couplingpairs'],
                 temp['dataset'], temp['sample_size'],
                 temp['method'],
                 temp['H'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
