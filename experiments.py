import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [5000]
    nf_couplingpairs = [2]
    no_hiddens = [16]


    ####################################################################################################################
    dataset = ['tanh']
    Hs = [121, 576, 1024]
    zeromeans = ['False']
    prior_params = ['0 1', '0 100']

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian'],
        'varparam0': ['1 1e-2'],
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
        'prior_param': prior_params,
        'method': ['nf_gamma'],
        'varparam0': ['100 1 100'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################################
    dataset = ['tanh']
    Hs = [121, 576, 1024]
    zeromeans = ['True']
    prior_params = ['5 1', '5 100']

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian'],
        'varparam0': ['1 1e-2'],
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
        'prior_param': prior_params,
        'method': ['nf_gamma'],
        'varparam0': ['100 1 100'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################################
    dataset = ['tanh']
    Hs = [121, 576, 1024]
    zeromeans = ['True']
    prior_params = ['5 1', '5 100']

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gamma'],
        'varparam0': ['100 1.25 100'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    ###################################################################################################################
    dataset = ['reducedrank']
    Hs = [6, 11, 24, 32]
    Hs = [11, 24, 32]
    zeromeans = ['False']
    prior_params = ['5 1', '5 100']

    hyperparameter_config = {
        'dataset': dataset,
        'H': Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian'],
        'varparam0': ['1 1e-2'],
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
        'prior_param': prior_params,
        'method': ['nf_gamma'],
        'varparam0': ['100 1 100', '100 0.25 100', '100 1.25 100'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

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
                 temp['prior_param'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
