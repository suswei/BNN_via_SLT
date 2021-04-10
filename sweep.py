import sys
import os
import itertools
import numpy as np


def set_sweep_config():

    tanh_Hs = [64, 81, 100]
    rr_Hs = [80, 90, 100]

    ns = [5000]
    seeds = [1, 2, 3, 4, 5]
    priors = ['gaussian']

    sweep_params = {'rr_Hs': rr_Hs, 'tanh_Hs': tanh_Hs, 'seeds': seeds}
    hyperparameter_experiments = []


    ####################################################################################################

    hyperparameter_config = {
        'H': tanh_Hs,
        'seed': seeds,
        'dataset': ['tanh'],
        'var_mode': ['nf_gammatrunc','nf_gaussian'],
        'n': ns,
        'prior': priors,
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

    return sweep_params, hyperparameter_experiments


def main(taskid):

    _, hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = '{}_{}_n{}_H{}_seed{}'.format(temp['var_mode'], temp['dataset'], temp['n'], temp['H'],temp['seed'])

    os.system("python3 main.py "
              "--dataset %s "
              "--epochs 5000 "
              "--prior %s "
              "--prior_var 1e-2 "
              "--sample_size %s "
              "--seed %s "
              "--H %s "
              "--var_mode %s "
              "--path %s"
              % (temp['dataset'], temp['prior'],
                 temp['n'], temp['seed'], temp['H'],
                 temp['var_mode'], path))


if __name__ == "__main__":
    main(sys.argv[1:])
