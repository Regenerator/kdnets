import theano.tensor as T

from lasagne.layers import Layer
from lasagne.init import Uniform, Constant


class SharedDotLayer(Layer):
    def __init__(self, incoming, num_units, W=Uniform(), b=Constant(0.), **kwargs):
        super(SharedDotLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        self.b = self.add_param(b, (num_units,), name='b', regularizable=False)
        
    def get_output_for(self, input, **kwargs):
        return T.tensordot(self.W, input, axes=(0, 1)).dimshuffle((1, 0, 2)) + self.b.dimshuffle(('x', 0, 'x'))
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units, input_shape[2])


class SPTNormReshapeLayer(Layer):
    def __init__(self, incoming, sign, axis, num_units, **kwargs):
        super(SPTNormReshapeLayer, self).__init__(incoming, **kwargs)
        self.sign = sign
        self.axis = axis
        self.num_units = num_units
    
    def get_output_for(self, input, **kwargs):
        if self.sign == '-':
            return T.tile(T.maximum(-input[:, self.axis, :].dimshuffle(0, 'x', 1), 0), (1, self.num_units, 1))
        elif self.sign == '+':
            return T.tile(T.maximum(input[:, self.axis, :].dimshuffle(0, 'x', 1), 0), (1, self.num_units, 1))
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units, input_shape[2])


class SPTUpscaleLayer(Layer):
    def __init__(self, incoming, mode='left', **kwargs):
        super(SPTUpscaleLayer, self).__init__(incoming, **kwargs)
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], 2*input_shape[2])

    def get_output_for(self, input, **kwargs):
        upscaled = input
        upscaled = T.zeros(shape=self.get_output_shape_for(input.shape), dtype=input.dtype)
        if self.mode == 'left':
            upscaled = T.set_subtensor(upscaled[:, :, ::2], input)
        elif self.mode == 'right':
            upscaled = T.set_subtensor(upscaled[:, :, 1::2], input)
        return upscaled
