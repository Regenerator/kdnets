import pickle

import theano.tensor as T
from theano import scan

from lasagne.layers import get_all_param_values, set_all_param_values


def dump_weights(filename, nn_output):
    with open(filename, 'wb') as paramsf:
        pickle.dump(get_all_param_values(nn_output), paramsf, pickle.HIGHEST_PROTOCOL)
        

def load_weights(filename, nn_output):
    with open(filename, 'rb') as paramsf:
        set_all_param_values(nn_output, pickle.load(paramsf))


def calc_hist_vals_vector(sim, hist_min, hist_max, sample_num, bin_num=256, min_cov=1e-6):
    sim_mat = T.tile(sim.reshape((sample_num, 1)), (1, bin_num))
    w = (hist_max - hist_min) / bin_num + min_cov
    grid_vals = T.arange(0, bin_num)*(hist_max-hist_min)/bin_num + hist_min + w/2.0
    grid = T.tile(grid_vals, (sample_num, 1))
    w_triang = 4.0*w + min_cov
    D = T._tensor_py_operators.__abs__(grid-sim_mat)
    mask = (D <= w_triang/2)
    D_fin = w_triang * (D*(-2.0 / w_triang ** 2) + 1.0 / w_triang)*mask
    hist_corr = T.sum(D_fin, 0)
    return hist_corr


def hist_loss(hist_neg, hist_pos, bin_num=256):
    agg_pos, _ = scan(fn = lambda ind, A: T.sum(A[0:ind+1]), outputs_info=None,
                      sequences=T.arange(bin_num), non_sequences=hist_pos)
    return T.sum(T.dot(agg_pos, hist_neg))
