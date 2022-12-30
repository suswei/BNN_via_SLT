import torch
import numpy as np
import torch.distributions as D

def get_lmbda_dim(Hs, dataset):
    """

    :param Hs: list of hidden units values
    :param dataset: string name
    :return: list of corresponding RLCTs
    """

    trueRLCT = []
    dim = []
    for H in Hs:
        if dataset == 'reducedrank':
            output_dim = H
            input_dim = output_dim + 3
            trueRLCT += [(output_dim * H - H ** 2 + input_dim * H) / 2]  # rank r = H
            dim += [input_dim * H + output_dim * H]

        elif dataset=='tanh':
            max_integer = int(H**(1/2))
            trueRLCT += [(H + max_integer * max_integer + max_integer) / (4 * max_integer + 2)]
            dim += [2*H]

    return trueRLCT, dim
