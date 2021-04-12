import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gammatrunc', 'nf_gaussian'],
        'H': [900, 1600, 6400],
        'sample_size': [5000, 10000, 20000]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gammatrunc'],
        'H': [30, 40, 80],
        'sample_size': [5000, 10000, 20000]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'varyn/{}_{}_n{}_H{}'.format(temp['method'], temp['dataset'], temp['sample_size'],temp['H'])

    os.system("python3 main.py "
              "--dataset %s "
              "--sample_size %s "
              "--H %s "
              "--method %s "
              "--path %s"
              % (temp['dataset'],
                 temp['sample_size'],
                 temp['H'],
                 temp['method'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
