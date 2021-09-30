import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [5000]
    nf_couplingpairs = [2]
    no_hiddens = [16]

    tanh_Hs = [576, 1024]
    rr_Hs = [24, 32]

    gaussian_varparam0 = ['0 1', '1 1e-2']
    gamma_varparam0 = ['100 0.5 100', '10 1 100', '10 1 1000', '500 1 100', '500 1 1000']

    ####################################################################################################################
    dataset = ['tanh']
    zeromeans = ['False']
    prior_params = ['0 1', '0 100']

    hyperparameter_config = {
        'dataset': dataset,
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian'],
        'varparam0': gaussian_varparam0,
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': dataset,
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gamma'],
        'varparam0': gamma_varparam0,
        'grad_flag': [True, False],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################################
    dataset = ['tanh']
    zeromeans = ['True']
    prior_params = ['5 1', '5 100']

    hyperparameter_config = {
        'dataset': dataset,
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian'],
        'varparam0': gaussian_varparam0,
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': dataset,
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gamma'],
        'varparam0': gamma_varparam0,
        'grad_flag': [True, False],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###################################################################################################################
    dataset = ['reducedrank']
    zeromeans = ['False']
    prior_params = ['5 1', '5 100']

    hyperparameter_config = {
        'dataset': dataset,
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian'],
        'varparam0': gaussian_varparam0,
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    hyperparameter_config = {
        'dataset': dataset,
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gamma'],
        'varparam0': gamma_varparam0,
        'grad_flag': [True, False],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments, tanh_Hs, rr_Hs


def main(taskid):

    hyperparameter_experiments, _, _ = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    # if not os.path.exists(path):
        # os.makedirs(path)

    torch.save(hyperparameter_experiments, '{}/hyp.pt'.format(temp['dataset']))

    path = '{}/taskid{}/'.format(temp['dataset'], taskid)
    os.system("python3 main.py "
              "--data %s %s %s %s "
              "--var_mode %s %s %s %s "
              "--epochs 2000 --display_interval 2000 "
              "--prior_dist gaussian %s "
              "--seeds 1 2 3 4 5 6 7 8 9 10 "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'], temp['zeromean'],
                 temp['method'], temp['nf_couplingpair'], temp['nf_hidden'], temp['varparam0'],
                 temp['prior_param'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
