import pickle
from lasagne.layers import get_all_param_values, set_all_param_values

def dump_weights(filename, nn_output):
    with open(filename, 'wb') as paramsf:
        pickle.dump(get_all_param_values(nn_output), paramsf, pickle.HIGHEST_PROTOCOL)
        

def load_weights(filename, nn_output):
    with open(filename, 'rb') as paramsf:
        set_all_param_values(nn_output, pickle.load(paramsf))
