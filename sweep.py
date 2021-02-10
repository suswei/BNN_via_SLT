import sys
import os
import itertools
from dataset_factory import get_lmbda


def set_sweep_config():

    rr_Hs = [80, 90, 100]
    rr_ls = get_lmbda(rr_Hs, 'reducedrank')

    tanh_Hs = [6400, 8100, 10000]
    tanh_ls = [1000, 2000, 4000]

    seeds = [1, 2, 3, 4, 5]

    sweep_params = {'rr_Hs': rr_Hs, 'tanh_Hs': tanh_Hs, 'rr_ls': rr_ls, 'tanh_ls': tanh_ls, 'seeds': seeds}
    hyperparameter_experiments = []

    ####################################################################################################

    hyperparameter_config = {
        'H': rr_Hs,
        'lmbda': rr_ls,
        'seed': seeds,
        'dataset': ['reducedrank'],
        'nf_hidden': [16],
        'nf_layers': [20],
        'mf_mode': ['gengamma']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ####################################################################################################

    hyperparameter_config = {
        'H': rr_Hs,
        'seed': seeds,
        'dataset': ['reducedrank'],
        'mf_mode': ['gaussian']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################

    hyperparameter_config = {
        'H': tanh_Hs,
        'lmbda': tanh_ls,
        'seed': seeds,
        'dataset': ['tanh'],
        'nf_hidden': [16],
        'nf_layers': [20],
        'mf_mode': ['gengamma']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ####################################################################################################

    hyperparameter_config = {
        'H': tanh_Hs,
        'seed': seeds,
        'dataset': ['tanh'],
        'mf_mode': ['gaussian']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return sweep_params, hyperparameter_experiments


def main(taskid):

    _, hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    if temp['mf_mode'] == 'gaussian':
        path = 'untransformed/untransformed_{}_H{}_seed{}'.format(temp['dataset'], temp['H'], temp['seed'])

        os.system("python3 main.py "
                  "--dataset %s "
                  "--seed %s "
                  "--H %s "
                  "--mf_mode gaussian "
                  "--path %s"
                  % (temp['dataset'], temp['seed'], temp['H'], path))

    else:

        path = 'transformed/transformed_{}_H{}_lmbda{}_seed{}'.format(temp['dataset'],temp['H'],round(temp['lmbda']),temp['seed'])
        os.system("python3 main.py "
                  "--dataset %s "
                  "--seed %s "
                  "--H %s "
                  "--lmbda_star %s "
                  "--nf_hidden %s "
                  "--nf_layers %s "
                  "--path %s"
                  %(temp['dataset'], temp['seed'], temp['H'], temp['lmbda'], temp['nf_hidden'],temp['nf_layers'], path))


if __name__ == "__main__":
    main(sys.argv[1:])
