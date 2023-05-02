import numpy as np

import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tfnp

import keras as K

from .common import *

class KerasReXNetV1(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        for n, m in model.named_children():
            setattr(self, n, eval("Keras" + m.__class__.__name__)(m, name = n))
        
    def call(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x

class KerasConvNormAct(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        self.conv = KerasConv2d(model.conv, name = "conv")
        self.bn = K.layers.BatchNormalization(momentum = 1 - model.bn.momentum, epsilon = model.bn.eps, name = "bn")
        self.act = eval("Keras" + model.bn.act.__class__.__name__)(model.bn.act, name = "act")

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class KerasSequential(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        self.models = []
        for i, m in enumerate(model):
            self.models.append(eval("Keras" + m.__class__.__name__)(m, name = str(i)))
    def call(self, x):
        for model in self.models:
            x = model(x)
        return x

class KerasLinearBottleneck(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        self.conv_exp = None
        self.se = None
        self.use_shortcut = model.use_shortcut
        self.in_channels = model.in_channels
        for n, m in model.named_children():
            setattr(self, n, eval("Keras" + m.__class__.__name__)(m, name = n))
        
    def call(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            x = tf.concat([x[...,0:self.in_channels] + shortcut, x[...,self.in_channels:]], -1)
        return x

class KerasSEModule(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        for n, m in model.named_children():
            setattr(self, n, eval("Keras" + m.__class__.__name__)(m, name = n))
        self.pool = K.layers.GlobalAveragePooling2D(keepdims = True)   

    def call(self, x):
        x_se = self.pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)

class KerasClassifierHead(K.layers.Layer):
    def __init__(self, model, name = None):
        super().__init__(name = name)
        for n, m in model.named_children():
            setattr(self, n, eval("Keras" + m.__class__.__name__)(m, name = n))
    def call(self, x):
        x = self.global_pool(x)
        x = self.fc(x)
        return self.flatten(x)