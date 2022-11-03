import sys
import os
import itertools
import torch


def set_sweep_config():

    hyperparameter_experiments = []

    sample_sizes = [500]
    nf_couplingpairs = [2]
    no_hiddens = [16]

    tanh_Hs = [100, 576]
    rr_Hs = [10, 24]

    ####################################################################################################################
    hyperparameter_config = {
        'dataset': ['tanh'],
        'H': tanh_Hs,
        'sample_size': sample_sizes,
        'zeromean': [True],
        'prior_param': ['0 1'],
        'base_dist': ['gengamma', 'gaussian_std', 'gaussian_match'],
        'grad_flag': ['True', 'False'],
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
        'base_dist': ['gengamma', 'gaussian_std', 'gaussian_match'],
        'grad_flag': ['True', 'False'],
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

    torch.save(hyperparameter_experiments, '{}/hyp.pt'.format(temp['dataset']))

    path = '{}/taskid{}/'.format(temp['dataset'], taskid)
    # if not os.path.exists(path):
    #     os.makedirs(path)

    os.system("python3 main.py "
              "--data %s %s %s 100 %s "
              "--var_mode %s %s %s %s "
              "--epochs 500 --display_interval 100 "
              "--seeds 1 2 3 4 5 "
              "--path %s "
              % (temp['dataset'], temp['H'], temp['sample_size'], temp['zeromean'],
                 temp['base_dist'], temp['nf_couplingpair'], temp['nf_hidden'], temp['grad_flag'],
                 path)
              )


if __name__ == "__main__":
    main(sys.argv[1:])
