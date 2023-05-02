import numpy as np
import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tfnp
import keras as K

import torch
import torch.nn as nn
import torch.nn.functional as F

#from ..augmentations import *

dis_idx0, dis_idx1 = np.where(np.triu(np.ones((21, 21)), 1) == 1)
dis_idx2, dis_idx3 = np.where(np.triu(np.ones((20, 20)), 1) == 1)


class KerasTransformerPreprocess(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.range = tf.range(0, config.max_position_embeddings, 1, dtype = tf.float32)
        self.flatten = K.layers.Flatten()
        self.max_position_embeddings = config.max_position_embeddings

    def cos_sim(self, A, B):
        n1 = tf.norm(A, axis = -1)
        n2 = tf.norm(B, axis = -1)
        sim = tfnp.sum(A * B, axis = -1) / n1 / n2
        return sim

    def norm(self, pos):
        ref = tf.reshape(pos, (-1,))
        ref = ref[~tfnp.isnan(ref)]
        mu = tfnp.mean(ref)
        std = tfnp.sqrt(tfnp.var(ref, ddof = 1, dtype = tf.float32))
        pos = (pos - mu) / std
        return pos

    def call(self, pos):
        end = tf.shape(pos)[0]
        step = tfnp.clip(self.max_position_embeddings, -1, end + 1)
        idx = tf.cast(self.range[:step - 1] * tf.cast(end, tf.float32) / tf.cast(step - 1, tf.float32), tf.int32)
        pos = tfnp.take(pos, idx, axis = 0)
        lip, lhand, rhand = tfnp.take(pos, LIP, axis = 1), pos[:,468:489], pos[:,522:543]
        rhand = tf.concat([2 * lip[:,0:1,0:1] - rhand[...,0:1], rhand[...,1:]], -1)
        lhand = tf.where(tfnp.sum(tfnp.isnan(lhand)) < tfnp.sum(tfnp.isnan(rhand)), lhand, rhand)

        ld = tf.norm(tfnp.take(lhand, dis_idx0, 1)[...,:2] - tfnp.take(lhand, dis_idx1, 1)[...,:2], axis = -1)
        lipd = tf.norm(tfnp.take(lip, dis_idx2, 1)[...,:2] - tfnp.take(lip, dis_idx3, 1)[...,:2], axis = -1)
        lsim = self.cos_sim(tfnp.take(lhand, HAND_ANGLES[:,0], 1) - tfnp.take(lhand, HAND_ANGLES[:,1], 1),
                            tfnp.take(lhand, HAND_ANGLES[:,2], 1) - tfnp.take(lhand, HAND_ANGLES[:,1], 1))
        lipsim = self.cos_sim(tfnp.take(lip, LIP_ANGLES[:,0], 1) - tfnp.take(lip, LIP_ANGLES[:,1], 1),
                              tfnp.take(lip, LIP_ANGLES[:,2], 1) - tfnp.take(lip, LIP_ANGLES[:,1], 1))

        pos = self.norm(tf.concat([lip, lhand], 1))
        offset = tf.zeros_like(pos[-1:])
        movement = pos[:-1] - pos[1:]
        dpos = tf.concat([movement, offset], 0)
        rdpos = tf.concat([offset, -movement], 0)

        pos = tf.concat([self.flatten(_) for _ in [pos, dpos, rdpos, lipd, ld, lipsim, lsim]], -1)
        pos = tf.where(tfnp.isnan(pos), 0.0, pos)
        return pos[None]


class KerasPreprocessing(K.layers.Layer):
    def __init__(self, size=(160, 80), upsample=True, name=None):
        super().__init__(name=name)
        self.interesting_idx = np.array([291, 37, 40, 267, 270, 80, 17, 82, 308, 181, 405, 375, 312, 310, 87,
                                         88, 317, 318, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
                                         479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 492, 493,
                                         494, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510,
                                         511, 512, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
                                         534, 535, 536, 537, 538, 539, 540, 541, 542])
        if upsample:
            self.upsample = K.layers.Resizing(*size, interpolation = "nearest")
        else:
            self.upsample = None
        
    def norm(self, pos):
        ref = tf.reshape(pos, (-1,))
        ref = ref[~tfnp.isnan(ref)]
        mu = tfnp.mean(ref)
        std = tfnp.sqrt(tfnp.var(ref, ddof = 1, dtype = tf.float32)) + 1e-6
        pos = (pos - mu) / std
        return pos
        
    def call(self, pos):
        pos = tfnp.take(pos, self.interesting_idx, 1)
        pos = self.norm(pos)
        pos = tf.where(tfnp.isnan(pos), 0.0, pos)
        if self.upsample is not None:
            pos = self.upsample(pos)
        return pos[None]


class Preprocessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_size = (160, 80)
        self.interesting_idx = np.array([291, 37, 40, 267, 270, 80, 17, 82, 308, 181, 405, 375, 312, 310, 87, 88, 317, 318, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 492, 493, 494, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542])
        
    @staticmethod
    def norm(pos):
        ref = pos.flatten()
        ref = ref[~ref.isnan()]
        mu, std = ref.mean(), ref.std() + 1e-6
        return (pos - mu) / std

    def forward(self, yy):
        yy = yy[:, self.interesting_idx, :]
        yy = self.norm(yy)
        yy = torch.where(torch.isnan(yy), torch.tensor(0.0, dtype=torch.float32).to(yy), yy)
        yy = F.interpolate(yy.permute(2, 0, 1)[None], size=self.inp_size, mode="nearest-exact")[0].permute(1, 2, 0)
        return yy.unsqueeze(0)