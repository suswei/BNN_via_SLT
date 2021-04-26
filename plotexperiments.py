import seaborn as sns
import torch
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from experiments import set_sweep_config
from scipy import stats
import numpy as np
from dataset_factory import get_lmbda

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='?')

    parser.add_argument('--dataset_of_interest', default='reducedrank', type=str, choices=['reducedrank', 'tanh'])

    parser.add_argument('--savefig', action='store_true')

    args = parser.parse_args()

    # if args.savefig:
    #     matplotlib.use("pgf")
    #     matplotlib.rcParams.update({
    #         "pgf.texsystem": "xelatex",
    #         'font.family': 'serif',
    #         'text.usetex': True,
    #         'pgf.rcfonts': False,
    #     })
    #     plt.rcParams["figure.figsize"] = (6.75/2, 3)

    prior_of_interest = 1e-2
    dataset_of_interest = 'tanh'
    path_prefix = 'highHlognslope'

    hyperparameter_experiments = torch.load('{}/hyp.pt'.format(path_prefix))
    tasks = hyperparameter_experiments.__len__()

    Hs_list = []
    method_list = []
    seed_list = []
    ev_list = []
    ns_list = []

    for taskid in range(tasks):

        path = '{}/taskid{}/'.format(path_prefix, taskid)
        try:
            results = torch.load('{}/results.pt'.format(path))
            sim_args = torch.load('{}/args.pt'.format(path))

            if prior_of_interest == 'unif':

                if sim_args['prior'] == prior_of_interest and sim_args['dataset'] == dataset_of_interest:

                    ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
                    # ev_list += [results['elbo_loglik_val'].detach().numpy()]
                    ns_list += [sim_args['sample_size']]

                    if sim_args['method'] == 'nf_gaussian':
                        method_list += [sim_args['method']]
                    else:
                        method_list += ['{}_{}'.format(sim_args['method'], sim_args['nf_gamma_mode'])]
                    seed_list += [sim_args['seed']]
                    Hs_list += [sim_args['H']]

            else:

                if sim_args['prior_var'] == prior_of_interest and sim_args['dataset'] == dataset_of_interest:

                    ev_list += [results['elbo'].detach().numpy() + sim_args['nSn'].numpy()]
                    # ev_list += [results['elbo_loglik_val'].detach().numpy()]
                    ns_list += [sim_args['sample_size']]

                    if sim_args['method'] == 'nf_gaussian':
                        method_list += [sim_args['method']]
                    else:
                        method_list += ['{}_{}'.format(sim_args['method'],sim_args['nf_gamma_mode'])]
                    seed_list += [sim_args['seed']]
                    Hs_list += [sim_args['H']]
        except:
            print('missing taskid {}'.format(taskid))

    unique_Hs = list(set(Hs_list))

    for H in unique_Hs:

        for taskid in range(tasks):

            try:
                path = '{}/taskid{}/'.format(path_prefix, taskid)
                results = torch.load('{}/results.pt'.format(path))
                sim_args = torch.load('{}/args.pt'.format(path))

                if sim_args['H'] == H:
                    ev_list += [results['asy_log_pDn']]
                    method_list += ['truth']
                    seed_list += [sim_args['seed']]
                    Hs_list += [H]
                    ns_list += [sim_args['sample_size']]

                    break

            except:
                print('missing taskid {}'.format(taskid))

    method_list = pd.Series(method_list, dtype='category')
    seed_list = pd.Series(seed_list, dtype="category")

    summary_pd = pd.DataFrame({'H': Hs_list,
                               'ELBOplusnSn': ev_list,
                               'n': ns_list,
                               'method': method_list,
                               'seed': seed_list})

    for H in [100, 900, 1600]:
        for method in ['nf_gamma_icml','nf_gaussian']:
            temp = summary_pd.loc[summary_pd['H'] == H]
            truth = get_lmbda([H], 'tanh')[0]
            temp = temp.loc[temp['method'] == method]
            evs = temp.groupby('n')['ELBOplusnSn'].mean()
            ns = [int(round(np.exp(4))) * 32, int(round(np.exp(5))) * 32, int(round(np.exp(6))) * 32,
                  int(round(np.exp(7))) * 32] # TODO: don't hardcode
            ns = [int(round(np.exp(4))) * 32, int(round(np.exp(5))) * 32, int(round(np.exp(6))) * 32] # TODO: don't hardcode
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(ns), evs.get_values())
            plt.plot(np.log(ns),evs.get_values(),'.')
            plt.title('H {}: method {}, \n truth {} versus slope {:2f} and R2 {:2f}'.format(H, method, -truth, slope, r_value))
            plt.show()

    #
    # g = sns.barplot(x="H", y="ELBOplusnSn",
    #                 hue="method",
    #                 data=summary_pd)
    #
    # sns.set_style("ticks")
    # hatches = ['/', '/', '/',
    #            '+', '+', '+',
    #            'x', 'x', 'x']
    # for hatch, patch in zip(hatches, g.patches):
    #     patch.set_hatch(hatch)
    # leg = plt.legend(bbox_to_anchor=(1, 1), loc=2)
    # for patch in leg.get_patches():
    #     patch.set_height(12)
    #     patch.set_y(-6)
    #
    # if prior_of_interest == 'unif':
    #     title = '{}/{}_unifprior.png'.format(path_prefix, dataset_of_interest)
    # else:
    #     title = '{}/{}_priorvar{}.png'.format(path_prefix, dataset_of_interest, prior_of_interest)
    # plt.title(title)
    #
    # if args.savefig:
    #         plt.savefig(title, bbox_inches='tight')
    # else:
    #     plt.show()
    #
    # plt.close()


if __name__ == "__main__":
    main()

