import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [5000]
    prior_vars = [1, 100]

    ####################################################################################################################

    nf_couplingpairs = [2]
    no_hiddens = [16]

    ####################################################################################################################
    dataset = ['tanh']
    Hs = [36, 121, 576, 1024]
    zeromeans = ['True']

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_var': prior_vars,
        'method': ['nf_gaussian'],
        'varparam0': ['1 5e-2', '1 5e-3'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_var': prior_vars,
        'method': ['nf_gamma'],
        'varparam0': ['100 0.5 100', '1000 0.5 1000'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################################
    # dataset = ['reducedrank']
    # Hs = [6, 11, 24, 32]
    # zeromeans = ['False']
    #
    # hyperparameter_config = {
    #     'dataset': dataset,
    #     'H': Hs,
    #     'sample_size': sample_sizes,
    #     'zeromean': zeromeans,
    #     'prior_var': prior_vars,
    #     'method': ['nf_gaussian'],
    #     'varparam0': ['0 1', '1 1e-3'],
    #     'nf_couplingpair': nf_couplingpairs,
    #     'nf_hidden': no_hiddens,
    #     'seed': seeds,
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]
    #
    # hyperparameter_config = {
    #     'dataset': dataset,
    #     'H': Hs,
    #     'sample_size': sample_sizes,
    #     'zeromean': zeromeans,
    #     'prior_var': prior_vars,
    #     'method': ['nf_gamma'],
    #     'varparam0': ['100 1', '1000 1'],
    #     'nf_couplingpair': nf_couplingpairs,
    #     'nf_hidden': no_hiddens,
    #     'seed': seeds,
    # }
    # keys, values = zip(*hyperparameter_config.items())
    # hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    # if not os.path.exists(path):
        # os.makedirs(path)

    torch.save(hyperparameter_experiments, '{}/hyp.pt'.format(temp['dataset']))

    path = '{}/taskid{}/'.format(temp['dataset'], taskid)
    os.system("python3 main.py "
              "--data %s %s %s %s "
              "--var_mode %s %s %s %s "
              "--epochs 1000 --display_interval 100 "
              "--prior_dist gaussian %s "
              "--seeds 1 2 3 4 5 "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'], temp['zeromean'],
                 temp['method'], temp['nf_couplingpair'], temp['nf_hidden'], temp['varparam0'],
                 temp['prior_var'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
