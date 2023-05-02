import numpy as np

import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tfnp

import keras as K

from .common import *

class KerasEfficientNet(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        for n, m in model.named_children():
            setattr(self, n, eval("Keras" + m.__class__.__name__)(m, name = n[1:]))
        self.flatten = K.layers.Flatten()
        
    def extract_features(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)
        x = self._blocks(x)
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)
        return x
        
    def call(self, x):
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = self.flatten(x)
        x = self._dropout(x)
        x = self._fc(x)
        return x
        
class KerasConv2dStaticSamePadding(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        if model.groups == model.in_channels:
            self.conv = KerasDepthwiseConv2D(model.kernel_size, strides = model.stride, use_bias = model.bias is not None) 
        else:
            self.conv = K.layers.Conv2D(model.out_channels, model.kernel_size, padding = "valid", strides = model.stride, dilation_rate = model.dilation, use_bias = model.bias is not None)
        for n, m in model.named_children():
            setattr(self, n, eval("Keras" + m.__class__.__name__)(m, name = n))
        
    def call(self, x):
        x = self.static_padding(x)
        x = self.conv(x)
        return x
    
            
class KerasMBConvBlock(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        for n, m in model.named_children():
            setattr(self, n, eval("Keras" + m.__class__.__name__)(m, name = n[1:]))
        self.expand_ratio = model._block_args.expand_ratio
        self.has_se = model.has_se
        self.input_filters, self.output_filters = model._block_args.input_filters, model._block_args.output_filters
        self.stride = model._block_args.stride
        self.id_skip = model.id_skip
        self.pool = K.layers.GlobalAveragePooling2D(keepdims = True)
        
    def call(self, x):
        inputs = x
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = self.pool(x)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = tf.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            x = x + inputs
        return x

class KerasModuleList(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        self.models = []
        for i, m in enumerate(model):
            self.models.append(eval("Keras" + m.__class__.__name__)(m, name = str(i)))
    def call(self, x):
        for model in self.models:
            x = model(x)
        return x