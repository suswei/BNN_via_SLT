import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []
    methods = ['nf_gamma']
    modes = ['allones', 'icml', 'exp', 'abs_gauss']

    seeds = [1]
    prior_vars = [1, 1e-1, 1e-2, 1e-4]

    tanh_Hs = [4, 16, 36]
    rr_Hs = [2, 4, 6]

    ############################################  GAUSSIAN PRIOR -- NF_GAMMA ########################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior': ['gaussian'],
        'prior_var': prior_vars,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': rr_Hs,
        'prior': ['gaussian'],
        'prior_var': prior_vars,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ################################################# UNIF PRIOR -- NF_GAMMA ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior': ['unif'],
        'prior_var': [0],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': rr_Hs,
        'prior': ['unif'],
        'prior_var': [0],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    #################################################  GAUSSIAN PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['na'],
        'H': tanh_Hs,
        'prior': ['gaussian'],
        'prior_var': prior_vars,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['na'],
        'H': rr_Hs,
        'prior': ['gaussian'],
        'prior_var': prior_vars,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ################################################# UNIF PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['na'],
        'H': tanh_Hs,
        'prior': ['unif'],
        'prior_var': [0],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['na'],
        'H': rr_Hs,
        'prior': ['unif'],
        'prior_var': [0],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'smallH'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--lmbda_star --beta_star --exact_EqLogq --epochs 3000 "
              "--nf_layers 1 --nf_af tanh "
              "--dataset %s "
              "--method %s "
              "--nf_gamma_mode %s "
              "--H %s "
              "--prior %s "
              "--prior_var %s "
              "--seed %s "
              "--path %s "
              % (temp['dataset'],
                 temp['method'],
                 temp['nf_gamma_mode'],
                 temp['H'],
                 temp['prior'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
