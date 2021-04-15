import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    methods = ['nf_gamma', 'nf_gammatrunc']
    seeds = [1, 2, 3, 4, 5]

    tanh_Hs = [64, 900, 1600, 6400]
    rr_Hs = [8, 10, 30, 40]

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'H': tanh_Hs,
        'prior_var': [1e-2, 1e-4],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': methods,
        'H': rr_Hs,
        'prior_var': [1, 1e-1, 1e-2, 1e-4],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'gaussprior/{}_{}_n5000_H{}_seed{}_prior{}' \
        .format(temp['method'], temp['dataset'], temp['H'], temp['seed'], temp['prior_var'])

    os.system("python3 main.py "
              "--dataset %s "
              "--seed %s "
              "--sample_size %s "
              "--H %s "
              "--method %s "
              "--prior %s "
              "--prior gaussian "
              "--path %s "
              % (temp['dataset'],
                 temp['seed'],
                 temp['sample_size'],
                 temp['H'],
                 temp['method'],
                 temp['prior_var'],
                 path
                 ))


if __name__ == "__main__":
    main(sys.argv[1:])
