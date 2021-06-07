import numpy as np
import scipy.sparse as sp
from scipy.special import comb
import torch

def get_lazy_rw_ith_features(features, k, pos, A_hat, beta):
    """ Get the features of Lazy Random Walk 
        res = C(k, pos)*beta^{k-pos}*{(1-beta)A_hat}^{pos} * features
        where beta is the lazy probability, and A_hat a normalized adj matrix 
    """
    temp = np.power(beta, (k-pos))
    for i in range(pos):
        features = torch.spmm((1-beta)*A_hat, features)
    return comb(k, pos)*temp*features

def get_lrw_pre_calculated_feature_list(features, A_hat, k, beta):
    """ Get [k+1] features, for itself and k-hop neighbors, using the decomposition of the vanilla GCN (Eq. 13)
        Input: 
            A_hat: a normalized adj matrix, like sysmetrical normalized or re-normalization trick in gcn
            beta: used in lazy random walk: beta*I + (1-beta)A_hat
                  --- when beta=0.5, this will reduce to the general (I+A_hat) gcn
                  --- beta>0 should be guaranteed, otherwise nodes will lose their self-node features 
                  --- maybe beta>0.5 should be better
        Return: featurelist[k+1]
    """
    featurelist = []
    for i in range(k+1):
        temp = get_lazy_rw_ith_features(features.clone().detach(), k, i, A_hat, beta)
        featurelist.append(temp) 
    return featurelist