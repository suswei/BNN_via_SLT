import sys
import os
import itertools
import numpy as np


def set_sweep_config():

    tanh_Hs = [6400]
    rr_Hs = [80]
    ns = [int(round(np.exp(4)))*20, int(round(np.exp(5)))*20, int(round(np.exp(6)))*20,
          int(round(np.exp(7)))*20]
    ns = [5000]
    seeds = [1]
    var_modes = ['nf_gamma', 'nf_gaussian']

    sweep_params = {'rr_Hs': rr_Hs, 'tanh_Hs': tanh_Hs, 'seeds': seeds}
    hyperparameter_experiments = []


    ####################################################################################################

    hyperparameter_config = {
        'H': tanh_Hs,
        'seed': seeds,
        'dataset': ['tanh'],
        'var_mode': var_modes,
        'n': ns
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ####################################################################################################

    hyperparameter_config = {
        'H': rr_Hs,
        'seed': seeds,
        'dataset': ['reducedrank'],
        'var_mode': var_modes,
        'n': ns
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return sweep_params, hyperparameter_experiments


def main(taskid):

    _, hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = '{}_{}_n{}_H{}_seed{}'.format(temp['var_mode'], temp['dataset'], temp['n'], temp['H'],temp['seed'])

    os.system("python3 main.py "
              "--dataset %s "
              "--epochs 5000 "
              "--prior gmm "
              "--sample_size %s "
              "--seed %s "
              "--H %s "
              "--var_mode %s "
              "--path %s"
              % (temp['dataset'], temp['n'], temp['seed'], temp['H'], temp['var_mode'], path))


if __name__ == "__main__":
    main(sys.argv[1:])
