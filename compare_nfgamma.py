import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []
    methods = ['nf_gamma', 'nf_gammatrunc']
    modes = ['allones', 'icml', 'exp', 'abs_gauss']
    seeds = [1, 2, 3, 4, 5]

    tanh_Hs = [64, 900, 1600, 6400]
    rr_Hs = [8, 10, 30, 40]
    ############################################ TANH GAUSSIAN PRIOR ########################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior_var': [1, 1e-1, 1e-2, 1e-4],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ############################################# RR GAUSSIAN PRIOR #######################################################

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': rr_Hs,
        'prior_var': [1, 1e-1, 1e-2, 1e-4],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ############################################## TANH UNIF PRIOR ######################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
        'prior': ['unif'],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ############################################## RR UNIF PRIOR ######################################################

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': rr_Hs,
        'prior': ['unif'],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    if taskid < 1280:

        path = 'gaussprior_nfgammamodes/{}_{}_n5000_H{}_seed{}_prior{}_varparams{}' \
            .format(temp['method'], temp['dataset'], temp['H'], temp['seed'],temp['prior_var'],temp['nf_gamma_mode'])

        os.system("python3 main.py "
                  "--dataset %s "
                  "--method %s "
                  "--nf_gamma_mode %s "
                  "--H %s "
                  "--prior gaussian "
                  "--prior_var %s "
                  "--seed %s "
                  "--path %s "
                  % (temp['dataset'],
                     temp['method'],
                     temp['nf_gamma_mode'],
                     temp['H'],
                     temp['prior_var'],
                     temp['seed'],
                     path))

    else:

        path = 'unifprior_nfgammamodes/{}_{}_n5000_H{}_seed{}_prior{}_varparams{}' \
            .format(temp['method'], temp['dataset'], temp['H'], temp['seed'], temp['prior'], temp['nf_gamma_mode'])

        os.system("python3 main.py "
                  "--dataset %s "
                  "--method %s "
                  "--nf_gamma_mode %s "
                  "--H %s "
                  "--prior unif "
                  "--seed %s "
                  "--path %s "
                  % (temp['dataset'],
                     temp['method'],
                     temp['nf_gamma_mode'],
                     temp['H'],
                     temp['seed'],
                     path))


if __name__ == "__main__":
    main(sys.argv[1:])
