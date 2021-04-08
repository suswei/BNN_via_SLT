import sys
import os
import itertools
import numpy as np


def set_sweep_config():

    tanh_Hs = [6400]
    rr_Hs = [80, 90, 100]
    ns = [5000]
    seeds = [1]
    rr_ls = [10, 100, 500]
    beta_modes = ['lmbda_star']
    priorvars = [1, 1e-2, 1e-4]

    sweep_params = {'rr_Hs': rr_Hs, 'tanh_Hs': tanh_Hs, 'seeds': seeds}
    hyperparameter_experiments = []


    ####################################################################################################

    # hyperparameter_config = {
    #     'H': tanh_Hs,
    #     'seed': seeds,
    #     'dataset': ['tanh'],
    #     'var_mode': var_modes,
    #     'n': ns
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ####################################################################################################

    hyperparameter_config = {
        'H': rr_Hs,
        'seed': seeds,
        'dataset': ['reducedrank'],
        'var_mode': ['nf_gamma'],
        'n': ns,
        'prior_var': priorvars,
        'lmbda_star': rr_ls,
        'beta_mode': beta_modes,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'H': rr_Hs,
        'seed': seeds,
        'dataset': ['reducedrank'],
        'var_mode': ['nf_gaussian'],
        'n': ns,
        'prior_var': priorvars,
        'lmbda_star': [20], #NA
        'beta_mode': ['ones'] #NA
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return sweep_params, hyperparameter_experiments


def main(taskid):

    _, hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    if temp['var_mode'] == 'nf_gaussian':
        path = '{}_{}_n{}_H{}_seed{}_priorvar{}'.format(temp['var_mode'], temp['dataset'], temp['n'], temp['H'],temp['seed'], temp['prior_var'])
    else:
        path = '{}_{}_n{}_H{}_seed{}_l{}_betamode{}_priorvar{}'.format(temp['var_mode'], temp['dataset'], temp['n'], temp['H'],temp['seed'],temp['lmbda_star'],temp['beta_mode'], temp['prior_var'])

    os.system("python3 main.py "
              "--dataset %s "
              "--epochs 5000 "
              "--prior gaussian "
              "--prior_var %s "
              "--sample_size %s "
              "--seed %s "
              "--H %s "
              "--var_mode %s "
              "--beta_mode %s "
              "--lmbda_star %s "
              "--path %s"
              % (temp['dataset'], temp['prior_var'], temp['n'], temp['seed'], temp['H'], temp['var_mode'], temp['beta_mode'], temp['lmbda_star'], path))


if __name__ == "__main__":
    main(sys.argv[1:])
