import numpy as np
import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tfnp
import keras as K

class KerasTransformerLinear(K.layers.Layer):
    def __init__(self, out_channel, use_bias = True, name = None):
        super().__init__(name = name)
        self.out_channel = out_channel
        self.use_bias = use_bias
        
    def build(self, input_shape):
        self.weight = self.add_weight(name = "kernel", shape = (input_shape[-1], self.out_channel))
        if self.use_bias:
            self.bias = self.add_weight(name = "bias", shape = (self.out_channel,))
        else:
            self.bias = None
            
    def call(self, x):
        x = tf.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

class KerasLinear(KerasTransformerLinear):
    def __init__(self, model, name = None):
        super().__init__(out_channel = model.out_features, use_bias = model.bias is not None, name = name)

class KerasZeroPad2d(K.layers.ZeroPadding2D):
    def __init__(self, model, name = None):
        super().__init__((model.padding[-2:], model.padding[:2]), name = name)
            
    
class KerasBatchNorm2d(K.layers.BatchNormalization):
    def __init__(self, model, name = None):
        super().__init__(momentum = 1 - model.momentum, epsilon = model.eps, name = name)

class KerasConv2d(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        if model.padding != (0, 0):
            self.pad = K.layers.ZeroPadding2D(model.padding)
        else:
            self.pad = None
        if model.groups == model.in_channels:
            self.conv = KerasDepthwiseConv2D(model.kernel_size, strides = model.stride, use_bias = model.bias is not None) 
        else:
            self.conv = K.layers.Conv2D(model.out_channels, model.kernel_size, padding = "valid", strides = model.stride, dilation_rate = model.dilation, use_bias = model.bias is not None)

    def call(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        return x


class KerasIdentity(K.layers.Layer):
    def __init__(self, model, name):
        super().__init__(name = name)
        
    def call(self, x):
        return x

    
class KerasDepthwiseConv2D(K.layers.Layer):
    def __init__(self, kernel_size = 3, strides = 1, use_bias = True, name = None):
        super().__init__(name = name)
        self.kernel_size = kernel_size[0]
        self.strides = strides[0]
        self.use_bias = use_bias
        
        
    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.weight = self.add_weight(name = "kernel", shape = (self.kernel_size, self.kernel_size, num_channels))
        self.height = input_shape[1] - self.kernel_size + 1
        self.width = input_shape[2] - self.kernel_size + 1
        if self.use_bias:
            self.bias = self.add_weight(name = "bias", shape = (num_channels,))
        else:
            self.bias = None
        
    def call(self, x):
        out = x[:,0:self.height:self.strides,0:self.width:self.strides] * self.weight[0,0]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == 0 and j == 0:
                    continue
                out += x[:,i:self.height + i:self.strides,j:self.width + j:self.strides] * self.weight[i,j]
        if self.bias is not None:
            out = out + self.bias
        return out

KerasSigmoid = lambda *args, **kwargs: tf.nn.sigmoid
KerasReLU = lambda *args, **kwargs: tf.nn.relu
KerasReLU6 = lambda *args, **kwargs: tf.nn.relu6
KerasSiLU = lambda *args, **kwargs: tf.nn.silu
KerasMemoryEfficientSwish = lambda *args, **kwargs: tf.nn.silu
KerasAdaptiveAvgPool2d = lambda *args, **kwargs: K.layers.GlobalAveragePooling2D(keepdims = True)
KerasSelectAdaptivePool2d = lambda *args, **kwargs: K.layers.GlobalAveragePooling2D(keepdims = False)
KerasDropout = KerasIdentity