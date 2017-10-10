from numpy import float32

import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')
import theano
import theano.tensor as T

from lasagne.layers import InputLayer, ReshapeLayer, NonlinearityLayer, ExpressionLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.layers import DenseLayer
from lasagne.layers.dnn import BatchNormDNNLayer
from lasagne.nonlinearities import rectify, softmax

from lasagne.layers import get_output, get_all_params
from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.updates import adam

from lib.nn.layers import SharedDotLayer, SPTNormReshapeLayer


def build_classification_network(config):
    steps = config.get('steps', 10)
    features = config.get('input_features', 'all')
    n_f = config.get('n_f', [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 128])
    n_output = config.get('n_output', 10)
    l2_reg = config.get('l2_regularization', 1e-3)
    learning_rate = config.get('learning_rate', 1e-3)

    
    clouds = T.tensor3(dtype='float32')
    norms = [T.tensor3(dtype='float32') for step in xrange(steps)]
    target = T.vector(dtype='int32')
    
    lr = theano.shared(float32(learning_rate))

    KDNet = {}
    if features == 'no':
        KDNet['input'] = InputLayer((None, 1, 2**steps), input_var=clouds)
    else:
        KDNet['input'] = InputLayer((None, 3, 2**steps), input_var=clouds)

    for i in xrange(steps):
        KDNet['norm{}_r'.format(i+1)] = InputLayer((None, 3, 2**(steps-1-i)), input_var=norms[i])
        KDNet['norm{}_l'.format(i+1)] = ExpressionLayer(KDNet['norm{}_r'.format(i+1)], lambda X: -X)

        KDNet['norm{}_l_X-'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i+1)], '-', 0, n_f[i+1])
        KDNet['norm{}_l_Y-'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i+1)], '-', 1, n_f[i+1])
        KDNet['norm{}_l_Z-'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i+1)], '-', 2, n_f[i+1])
        KDNet['norm{}_l_X+'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i+1)], '+', 0, n_f[i+1])
        KDNet['norm{}_l_Y+'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i+1)], '+', 1, n_f[i+1])
        KDNet['norm{}_l_Z+'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_l'.format(i+1)], '+', 2, n_f[i+1])
    
        KDNet['norm{}_r_X-'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i+1)], '-', 0, n_f[i+1])
        KDNet['norm{}_r_Y-'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i+1)], '-', 1, n_f[i+1])
        KDNet['norm{}_r_Z-'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i+1)], '-', 2, n_f[i+1])
        KDNet['norm{}_r_X+'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i+1)], '+', 0, n_f[i+1])
        KDNet['norm{}_r_Y+'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i+1)], '+', 1, n_f[i+1])
        KDNet['norm{}_r_Z+'.format(i+1)] = SPTNormReshapeLayer(KDNet['norm{}_r'.format(i+1)], '+', 2, n_f[i+1])

        KDNet['cloud{}'.format(i+1)] = SharedDotLayer(KDNet['input'], n_f[i]) if i == 0 else \
                                        ElemwiseSumLayer([KDNet['cloud{}_l_X-_masked'.format(i)],
                                                          KDNet['cloud{}_l_Y-_masked'.format(i)],
                                                          KDNet['cloud{}_l_Z-_masked'.format(i)],
                                                          KDNet['cloud{}_l_X+_masked'.format(i)],
                                                          KDNet['cloud{}_l_Y+_masked'.format(i)],
                                                          KDNet['cloud{}_l_Z+_masked'.format(i)],
                                                          KDNet['cloud{}_r_X-_masked'.format(i)],
                                                          KDNet['cloud{}_r_Y-_masked'.format(i)],
                                                          KDNet['cloud{}_r_Z-_masked'.format(i)],
                                                          KDNet['cloud{}_r_X+_masked'.format(i)],
                                                          KDNet['cloud{}_r_Y+_masked'.format(i)],
                                                          KDNet['cloud{}_r_Z+_masked'.format(i)]])
        KDNet['cloud{}_bn'.format(i+1)] = BatchNormDNNLayer(KDNet['cloud{}'.format(i+1)])
        KDNet['cloud{}_relu'.format(i+1)] = NonlinearityLayer(KDNet['cloud{}_bn'.format(i+1)], rectify)

        KDNet['cloud{}_r'.format(i+1)] = ExpressionLayer(KDNet['cloud{}_relu'.format(i+1)],
                                                          lambda X: X[:, :, 1::2], (None, n_f[i], 2**(steps - i - 1)))
        KDNet['cloud{}_l'.format(i+1)] = ExpressionLayer(KDNet['cloud{}_relu'.format(i+1)],
                                                          lambda X: X[:, :, ::2], (None, n_f[i], 2**(steps - i - 1)))
        
        KDNet['cloud{}_l_X-'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i+1)], n_f[i+1])
        KDNet['cloud{}_l_Y-'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i+1)], n_f[i+1])
        KDNet['cloud{}_l_Z-'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i+1)], n_f[i+1])
        KDNet['cloud{}_l_X+'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i+1)], n_f[i+1])
        KDNet['cloud{}_l_Y+'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i+1)], n_f[i+1])
        KDNet['cloud{}_l_Z+'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_l'.format(i+1)], n_f[i+1])
        
        KDNet['cloud{}_r_X-'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i+1)], n_f[i+1],
                                                           W=KDNet['cloud{}_l_X-'.format(i+1)].W, 
                                                           b=KDNet['cloud{}_l_X-'.format(i+1)].b)
        KDNet['cloud{}_r_Y-'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i+1)], n_f[i+1],
                                                           W=KDNet['cloud{}_l_Y-'.format(i+1)].W, 
                                                           b=KDNet['cloud{}_l_Y-'.format(i+1)].b)
        KDNet['cloud{}_r_Z-'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i+1)], n_f[i+1],
                                                           W=KDNet['cloud{}_l_Z-'.format(i+1)].W, 
                                                           b=KDNet['cloud{}_l_Z-'.format(i+1)].b)
        KDNet['cloud{}_r_X+'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i+1)], n_f[i+1],
                                                           W=KDNet['cloud{}_l_X+'.format(i+1)].W,
                                                           b=KDNet['cloud{}_l_X+'.format(i+1)].b)
        KDNet['cloud{}_r_Y+'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i+1)], n_f[i+1],
                                                           W=KDNet['cloud{}_l_Y+'.format(i+1)].W,
                                                           b=KDNet['cloud{}_l_Y+'.format(i+1)].b)
        KDNet['cloud{}_r_Z+'.format(i+1)] = SharedDotLayer(KDNet['cloud{}_r'.format(i+1)], n_f[i+1],
                                                           W=KDNet['cloud{}_l_Z+'.format(i+1)].W,
                                                           b=KDNet['cloud{}_l_Z+'.format(i+1)].b)

        KDNet['cloud{}_l_X-_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_X-'.format(i+1)],
                                                                        KDNet['norm{}_l_X-'.format(i+1)]], T.mul)
        KDNet['cloud{}_l_Y-_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Y-'.format(i+1)],
                                                                        KDNet['norm{}_l_Y-'.format(i+1)]], T.mul)
        KDNet['cloud{}_l_Z-_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Z-'.format(i+1)],
                                                                        KDNet['norm{}_l_Z-'.format(i+1)]], T.mul)
        KDNet['cloud{}_l_X+_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_X+'.format(i+1)],
                                                                        KDNet['norm{}_l_X+'.format(i+1)]], T.mul)
        KDNet['cloud{}_l_Y+_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Y+'.format(i+1)],
                                                                        KDNet['norm{}_l_Y+'.format(i+1)]], T.mul)
        KDNet['cloud{}_l_Z+_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_l_Z+'.format(i+1)],
                                                                        KDNet['norm{}_l_Z+'.format(i+1)]], T.mul)
        KDNet['cloud{}_r_X-_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_X-'.format(i+1)],
                                                                        KDNet['norm{}_r_X-'.format(i+1)]], T.mul)
        KDNet['cloud{}_r_Y-_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Y-'.format(i+1)],
                                                                        KDNet['norm{}_r_Y-'.format(i+1)]], T.mul)
        KDNet['cloud{}_r_Z-_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Z-'.format(i+1)],
                                                                        KDNet['norm{}_r_Z-'.format(i+1)]], T.mul)
        KDNet['cloud{}_r_X+_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_X+'.format(i+1)],
                                                                        KDNet['norm{}_r_X+'.format(i+1)]], T.mul)
        KDNet['cloud{}_r_Y+_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Y+'.format(i+1)],
                                                                        KDNet['norm{}_r_Y+'.format(i+1)]], T.mul)
        KDNet['cloud{}_r_Z+_masked'.format(i+1)] = ElemwiseMergeLayer([KDNet['cloud{}_r_Z+'.format(i+1)],
                                                                        KDNet['norm{}_r_Z+'.format(i+1)]], T.mul)

    KDNet['cloud_fin'] = ElemwiseSumLayer([KDNet['cloud{}_l_X-_masked'.format(steps)],
                                           KDNet['cloud{}_l_Y-_masked'.format(steps)],
                                           KDNet['cloud{}_l_Z-_masked'.format(steps)],
                                           KDNet['cloud{}_l_X+_masked'.format(steps)],
                                           KDNet['cloud{}_l_Y+_masked'.format(steps)],
                                           KDNet['cloud{}_l_Z+_masked'.format(steps)],
                                           KDNet['cloud{}_r_X-_masked'.format(steps)],
                                           KDNet['cloud{}_r_Y-_masked'.format(steps)],
                                           KDNet['cloud{}_r_Z-_masked'.format(steps)],
                                           KDNet['cloud{}_r_X+_masked'.format(steps)],
                                           KDNet['cloud{}_r_Y+_masked'.format(steps)],
                                           KDNet['cloud{}_r_Z+_masked'.format(steps)]])
    KDNet['cloud_fin_bn'] = BatchNormDNNLayer(KDNet['cloud_fin'])
    KDNet['cloud_fin_relu'] = NonlinearityLayer(KDNet['cloud_fin_bn'], rectify)
    KDNet['cloud_fin_reshape'] = ReshapeLayer(KDNet['cloud_fin_relu'], (-1, n_f[-1]))
    KDNet['output'] = DenseLayer(KDNet['cloud_fin_reshape'], n_output, nonlinearity=softmax)

    prob = get_output(KDNet['output'])
    prob_det = get_output(KDNet['output'], deterministic=True)
    
    weights = get_all_params(KDNet['output'], trainable=True)
    l2_pen = regularize_network_params(KDNet['output'], l2)
    
    loss = categorical_crossentropy(prob, target).mean() + l2_reg*l2_pen
    accuracy = categorical_accuracy(prob, target).mean()
    
    updates = adam(loss, weights, learning_rate=lr)
    
    train_fun = theano.function([clouds] + norms + [target], [loss, accuracy], updates=updates)
    prob_fun = theano.function([clouds] + norms, prob_det)
    
    return train_fun, prob_fun, KDNet, lr
