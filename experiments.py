import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    nfs = ['2 4', '2 16', '4 4', '4 16']
    lrs = [0.001, 0.01]

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh_zeromean', 'tanh'],
        'H': [15, 115, 280],
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
        'H': [2, 16, 40],
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
        'H': [3, 10, 16],
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
              "--ns 2880 4736 7840 "
              "--var_mode %s %s %s "
              "--trainR 10 "
              "--lr %s "
              "--epochs 2000 " 
              "--display_interval 200 "
              "--seeds 1 2 3 4 5 6 7 8 9 10 "
              "--path %s "
              % (temp['dataset'], temp['H'],
                 temp['base_dist'], temp['nf'], temp['grad_flag'], temp['lr'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
