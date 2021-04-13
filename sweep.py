import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        # 'method': ['nf_gammatrunc','nf_gamma'],
        # 'varparams_mode': ['allones', 'icml', 'abs_gauss_n','exp'],
        'method': ['nf_gamma'],
        'H': [64, 100, 900, 1600],
        'sample_size': [5000],
        'prior_var': [1e-1, 1e-2],
        'seed': [1,2,3,4,5]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ####################################################################################################

    hyperparameter_config = {
        'dataset': ['reducedrank'],
        # 'method': ['nf_gammatrunc', 'nf_gamma'],
        # 'varparams_mode': ['allones', 'icml', 'abs_gauss_n','exp'],
        'method': ['nf_gamma'],
        'H': [8, 10, 30, 40],
        'sample_size': [5000],
        'prior_var': [1e-1, 1e-2],
        'seed': [1, 2, 3, 4, 5]
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    return hyperparameter_experiments


def main(taskid):

    hyperparameter_experiments = set_sweep_config()
    taskid = int(taskid[0])
    temp = hyperparameter_experiments[taskid]

    # path = 'nfgamma_comp/{}_{}_n{}_H{}_prior{}_seed{}_varparams{}'\
    #     .format(temp['method'], temp['dataset'], temp['sample_size'],temp['H'],temp['prior_var'],temp['seed'],temp['varparams_mode'])

    path = 'nfgamma_iaf/{}_{}_n{}_H{}_prior{}_seed{}' \
        .format(temp['method'], temp['dataset'], temp['sample_size'], temp['H'], temp['prior_var'], temp['seed'])

    os.system("python3 main.py "
              "--nf iaf "
              "--dataset %s "
              "--seed %s "
              "--sample_size %s "
              "--H %s "
              "--method %s "
              "--prior_var %s "
              "--epochs 2000 "
              "--path %s "
              # "--varparams_mode %s"
              % (temp['dataset'],
                 temp['seed'],
                 temp['sample_size'],
                 temp['H'],
                 temp['method'],
                 temp['prior_var'],
                 path
                 # temp['varparams_mode']
                 ))


if __name__ == "__main__":
    main(sys.argv[1:])
