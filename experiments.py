import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    nfs = ['1 4', '2 16']
    lrs = [0.01]
    seeds = list(range(1, 31))

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh_zeromean', 'tanh'],
        'H': [15, 50, 115],
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':  lrs,
        'nf': nfs,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'H': [2, 7, 10],
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':   lrs,
        'nf': nfs,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    hyperparameter_config = {
        'dataset': ['ffrelu'],
        'H': [3, 7, 16],
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':  lrs,
        'nf': nfs,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'results/taskid{}/'.format(taskid)

    os.system("python3 main.py "
              "--data %s %s "
              # "--ns 1760 2880 4736 7840 10048 "            
              "--ns 200 239 285 341 408 489 584 699 836 1000 "
              "--var_mode %s %s %s "
              "--lr %s "
              "--display_interval 1000 "
              "--seed %s "
              "--path %s "
              % (temp['dataset'], temp['H'],
                 temp['base_dist'], temp['nf'], temp['grad_flag'], temp['lr'],
                 temp['seed'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
