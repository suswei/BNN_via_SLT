import sys
import os
import itertools
import numpy as np


def set_sweep_config():

    hyperparameter_experiments = []

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'n': [5000],
        'var_mode': ['nf_gammatrunc'],
        'varparams_mode': ['abs_gauss','icml'],
        'H': [64, 81, 100],
        'seed': [1, 2, 3, 4, 5],
        'prior': ['unif','gaussian'],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['tanh'],
        'n': [5000],
        'var_mode': ['nf_gaussian'],
        'varparams_mode': ['abs_gauss'],  # NA
        'H': [64, 81, 100],
        'seed': [1, 2, 3, 4, 5],
        'prior': ['unif', 'gaussian'],
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ####################################################################################################

    # hyperparameter_config = {
    #     'H': rr_Hs,
    #     'seed': seeds,
    #     'dataset': ['reducedrank'],
    #     'var_mode': ['nf_gammatrunc','nf_gaussian'],
    #     'n': ns,
    #     'xi_upper': [4],
    #     'prior': priors,
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = '{}_{}_n{}_H{}_seed{}_prior{}_varparams{}'.format(temp['var_mode'], temp['dataset'], temp['n'], temp['H'],temp['seed'],temp['prior'], temp['varparams_mode'])

    os.system("python3 main.py "
              "--dataset %s "
              "--epochs 2000 "
              "--prior %s "
              "--prior_var 1e-1 "
              "--sample_size %s "
              "--seed %s "
              "--H %s "
              "--var_mode %s "
              "--varparams_mode %s "
              "--path %s"
              % (temp['dataset'], temp['prior'],
                 temp['n'], temp['seed'], temp['H'],
                 temp['var_mode'], temp['varparams_mode'], path))


if __name__ == "__main__":
    main(sys.argv[1:])
