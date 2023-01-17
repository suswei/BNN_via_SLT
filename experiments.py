import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    nfs = ['1 4']
    lrs = [0.001]

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh_zeromean', 'tanh'],
        'H': [15],
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':  lrs,
        'nf': nfs
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'H': [2],
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':   lrs,
        'nf': nfs
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    hyperparameter_config = {
        'dataset': ['ffrelu'],
        'H': [3],
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian'],
        'grad_flag': [False],
        'lr':  lrs,
        'nf': nfs
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
              # "--ns 2880 4736 7840 "
              "--ns 200 239 285 341 408 489 584 699 836 1000 "
              "--var_mode %s %s %s "
              "--trainR 10 "
              "--lr %s "
              "--epochs 1000 " 
              "--display_interval 1000 "
              "--seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 "
              "--path %s "
              % (temp['dataset'], temp['H'],
                 temp['base_dist'], temp['nf'], temp['grad_flag'], temp['lr'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
