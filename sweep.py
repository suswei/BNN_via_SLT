import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gammatrunc'],
        'H': [6400],
        'seed': [1, 2, 3, 4, 5],
        'nf_layer': [20]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gaussian'],
        'H': [64, 100, 6400, 10000],
        'seed': [1, 2, 3, 4, 5],
        'nf_layer': [50]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gammatrunc'],
        'H': [8, 10, 80, 100],
        'seed': [1, 2, 3, 4, 5],
        'nf_layer': [20]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gaussian'],
        'H': [8, 10, 80, 100],
        'seed': [1, 2, 3, 4, 5],
        'nf_layer': [20]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = '{}_{}_H{}_seed{}'.format(temp['method'], temp['dataset'], temp['H'], temp['seed'])

    os.system("python3 main.py "
              "--epochs 5000 "
              "--dataset %s "
              "--seed %s "
              "--H %s "
              "--method %s "
              "--nf_layer %s "
              "--path %s"
              % (temp['dataset'],
                 temp['seed'],
                 temp['H'],
                 temp['method'],
                 temp['nf_layer'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
