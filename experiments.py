import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []
    methods = ['nf_gamma']
    modes = ['icml', 'allones']

    seeds = [1, 2, 3, 4, 5]
    prior_vars = [1e-2]

    tanh_Hs = [1600, 6400]
    rr_Hs = [40, 80]

    ############################################  GAUSSIAN PRIOR -- NF_GAMMA ########################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior': ['gaussian'],
        'prior_var': prior_vars,
        'seed': seeds,
        'zeromean': ['True','False']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': methods,
    #     'nf_gamma_mode': modes,
    #     'H': rr_Hs,
    #     'prior': ['gaussian'],
    #     'prior_var': prior_vars,
    #     'seed': seeds
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ################################################# UNIF PRIOR -- NF_GAMMA ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior': ['unif'],
        'prior_var': [0],
        'seed': seeds,
        'zeromean': ['True', 'False']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    #
    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': methods,
    #     'nf_gamma_mode': modes,
    #     'H': rr_Hs,
    #     'prior': ['unif'],
    #     'prior_var': [0],
    #     'seed': seeds
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    #################################################  GAUSSIAN PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['icml'],
        'H': tanh_Hs,
        'prior': ['gaussian'],
        'prior_var': prior_vars,
        'seed': seeds,
        'zeromean': ['True', 'False']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': ['nf_gaussian'],
    #     'nf_gamma_mode': ['icml'],
    #     'H': rr_Hs,
    #     'prior': ['gaussian'],
    #     'prior_var': prior_vars,
    #     'seed': seeds
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ################################################# UNIF PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gaussian'],
        'nf_gamma_mode': ['icml'],
        'H': tanh_Hs,
        'prior': ['unif'],
        'prior_var': [0],
        'seed': seeds,
        'zeromean': ['True', 'False']
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    #
    # hyperparameter_config = {
    #     'dataset': ['reducedrank'],
    #     'method': ['nf_gaussian'],
    #     'nf_gamma_mode': ['icml'],
    #     'H': rr_Hs,
    #     'prior': ['unif'],
    #     'prior_var': [0],
    #     'seed': seeds
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    path = 'highHnvp'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(hyperparameter_experiments,'{}/hyp.pt'.format(path))

    path = '{}/taskid{}/'.format(path,taskid)

    os.system("python3 main.py "
              "--lmbda_star --beta_star --exact_EqLogq --epochs 2000 --trainR 1 "
              "--nf_layers 20 --nf_af tanh "
              "--dataset %s --zeromean %s "
              "--method %s "
              "--nf_gamma_mode %s "
              "--H %s "
              "--prior %s "
              "--prior_var %s "
              "--seed %s "
              "--path %s "
              % (temp['dataset'], temp['zeromean'],
                 temp['method'],
                 temp['nf_gamma_mode'],
                 temp['H'],
                 temp['prior'],
                 temp['prior_var'],
                 temp['seed'],
                 path))


if __name__ == "__main__":
    main(sys.argv[1:])
