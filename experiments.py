import sys
import os
import itertools


def set_sweep_config():

    hyperparameter_experiments = []
    methods = ['nf_gamma', 'nf_gammatrunc']
    modes = ['allones', 'icml', 'exp', 'abs_gauss']

    seeds = [1]
    prior_vars = [1, 1e-1, 1e-2, 1e-4]

    tanh_Hs = [64, 900, 1600]
    rr_Hs = [8, 30, 40]

    ############################################  GAUSSIAN PRIOR -- NF_GAMMA ########################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': methods,
        'nf_gamma_mode': modes,
        'H': tanh_Hs,
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
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    #################################################  GAUSSIAN PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gaussian'],
        'H': tanh_Hs,
        'prior_var': prior_vars,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gaussian'],
        'H': rr_Hs,
        'prior_var': prior_vars,
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]

    ################################################# UNIF PRIOR -- NF_GAUSSIAN ###################################################

    hyperparameter_config = {
        'dataset': ['tanh'],
        'method': ['nf_gaussian'],
        'H': tanh_Hs,
        'prior': ['unif'],
        'seed': seeds
    }
    keys, values = zip(*hyperparameter_config.items())
    hyperparameter_experiments += [dict(zip(keys, v)) for v in itertools.product(*values)]


    hyperparameter_config = {
        'dataset': ['reducedrank'],
        'method': ['nf_gaussian'],
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

    if taskid < 192:

        path = 'taskid{}/{}_{}_H{}_seed{}_prior{}_varparams{}' \
            .format(taskid, temp['method'], temp['dataset'], temp['H'], temp['seed'], temp['prior_var'], temp['nf_gamma_mode'])

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

    elif taskid < 240:

        path = 'taskid{}/{}_{}_H{}_seed{}_varparams{}' \
            .format(taskid, temp['method'], temp['dataset'], temp['H'], temp['seed'], temp['nf_gamma_mode'])

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

    elif taskid < 264:
        path = 'taskid{}/{}_{}_H{}_seed{}_prior{}' \
            .format(taskid, temp['method'], temp['dataset'], temp['H'], temp['seed'], temp['prior_var'])

        os.system("python3 main.py "
                  "--dataset %s "
                  "--seed %s "
                  "--H %s "
                  "--method %s "
                  "--prior %s "
                  "--prior gaussian "
                  "--path %s "
                  % (temp['dataset'],
                     temp['seed'],
                     temp['H'],
                     temp['method'],
                     temp['prior_var'],
                     path
                     ))
    else:

        path = 'taskid{}/{}_{}_H{}_seed{}' \
            .format(taskid, temp['method'], temp['dataset'], temp['H'], temp['seed'])

        os.system("python3 main.py "
                  "--dataset %s "
                  "--seed %s "
                  "--H %s "
                  "--method %s "
                  "--prior unif "
                  "--path %s "
                  % (temp['dataset'],
                     temp['seed'],
                     temp['H'],
                     temp['method'],
                     path
                     ))


if __name__ == "__main__":
    main(sys.argv[1:])
