import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [5000]
    nf_couplingpairs = [2]
    no_hiddens = [16]

    tanh_Hs = [16, 25]
    rr_Hs = [24, 32]

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'zeromean': [True],
        'prior_param': ['0 1'],
        'base_dist': ['gengammatrunc', 'gaussian'],
        'grad_flag': [True, False],
        'nf_couplingpair': nf_couplingpairs,
        'nf_hidden': no_hiddens,
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ###############################################################################
    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'H': rr_Hs,
        'sample_size': sample_sizes,
        'zeromean': [True],
        'prior_param': ['0 1'],
        'base_dist': ['gengammatrunc', 'gaussian'],
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
              "--var_mode %s %s %s "
              "--grad_flag %s "
              "--epochs 200 --display_interval 200 "
              "--seeds 1 "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'], temp['zeromean'],
                 temp['base_dist'], temp['nf_couplingpair'], temp['nf_hidden'],
                 temp['grad_flag'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
