import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [5000]
    nf_couplingpairs = [2]
    no_hiddens = [16]

    tanh_Hs = [20, 40]
    rr_Hs = [2, 4]


    ####################################################################################################################

    dataset = ['tanh']
    zeromeans = ['True']
    prior_params = ['0 1']

    hyperparameter_config = {
        'dataset': dataset,
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian', 'nf_gammatrunc'],
        'grad_flag': ['True'],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    dataset = ['reducedrank']
    zeromeans = ['True']
    prior_params = ['0 1']

    hyperparameter_config = {
        'dataset': dataset,
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'zeromean': zeromeans,
        'prior_param': prior_params,
        'method': ['nf_gaussian','nf_gammatrunc'],
        'grad_flag': ['True'],
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
        'grad_flag': ['True'],
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
              "--var_mode %s %s %s "
              "--grad_flag %s "
              "--epochs 2000 --display_interval 10 "
              "--prior_dist gaussian %s "
              "--seeds 1 2 3 4 5 6 7 8 9 10 "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'], temp['zeromean'],
                 temp['method'], temp['nf_couplingpair'], temp['nf_hidden'],
                 temp['grad_flag'],
                 temp['prior_param'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
