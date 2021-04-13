import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gammatrunc', 'nf_gaussian'],
        'H': [64, 100, 900, 1600, 6400, 10000],
        'sample_size': [5000, 10000, 20000],
        'prior_var': [1e-1, 1e-4],
        'seed': [1,2,3,4,5]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gammatrunc','nf_gaussian'],
        'H': [8, 10, 30, 40, 80, 100],
        'sample_size': [5000, 10000, 20000],
        'prior_var': [1e-1, 1e-4],
        'seed': [1, 2, 3, 4, 5]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'nfgaussian_comp/{}_{}_n{}_H{}_prior{}_seed{}'\
        .format(temp['method'], temp['dataset'], temp['sample_size'],temp['H'],temp['prior_var'],temp['seed'])

    os.system("python3 main.py "
              "--dataset %s "
              "--seed %s "
              "--sample_size %s "
              "--H %s "
              "--method %s "
              "--prior_var %s "
              "--epochs 5000 "
              "--path %s"
              % (temp['dataset'],
                 temp['seed'],
                 temp['sample_size'],
                 temp['H'],
                 temp['method'],
                 temp['prior_var'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
