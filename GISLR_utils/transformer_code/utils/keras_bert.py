import tensorflow as tf
import tensorflow.experimental.numpy as tfnp
import tensorflow.keras as K
from .augmentations import *

dis_idx0, dis_idx1 = np.where(np.triu(np.ones((21, 21)), 1) == 1)
dis_idx2, dis_idx3 = np.where(np.triu(np.ones((20, 20)), 1) == 1)


class Preprocess(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.range = tf.range(0, config.max_position_embeddings, 1, dtype = tf.float32)
        self.flatten = K.layers.Flatten()
#         self.LIP = tf.convert_to_tensor(LIP)
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

class TFBertEmbedding(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.position_embeddings = K.layers.Embedding(config.max_position_embeddings, config.hidden_size, name = "position_embeddings")
        self.token_type_embeddings = K.layers.Embedding(2, config.hidden_size, name = "token_type_embeddings")
        self.position_ids = tf.range(config.max_position_embeddings)
        self.token_type_ids = tf.zeros(config.max_position_embeddings)
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
#         self.dropout = K.layers.Dropout(config.hidden_dropout_prob)
        
    def call(self, inputs_embeds):
        position_ids = self.position_ids[:tf.shape(inputs_embeds)[1]]
        token_type_ids = self.token_type_ids[:tf.shape(inputs_embeds)[1]]
        embeddings = inputs_embeds 
        embeddings += self.position_embeddings(position_ids)
        embeddings += self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
        return embeddings

class TFBertEncoderCPE(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.layer = [
            TFBertLayerCPE(config, _, name = f"layer.{_}") 
#             Linear(256)
        for _ in range(config.num_hidden_layers)]
        
    def call(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)
        return hidden_states
    
class TFBertLayerCPE(K.layers.Layer):
    def __init__(self, config, layer_idx, name = None):
        super().__init__(name = name)
        self.attention = TFBertAttention(config, name = "attention")
        self.intermediate = TFIntermediate(config, name = "intermediate")
        self.output_ = TFOutput(config, name = "output")
        
        if config.cpe_start <= layer_idx < config.cpe_end:
            self.cpe_conv = K.layers.Conv1D(config.hidden_size, config.cpe_kernel_size, 1, padding = "same", groups = config.hidden_size, name = "cpe_conv")
        else:
            self.cpe_conv = None
        
    def cpe(self, hidden_states):
        if self.cpe_conv is not None:
            hidden_states += self.cpe_conv(hidden_states)
        return hidden_states
    
    def call(self, hidden_states):
        hidden_states = self.cpe(hidden_states)
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output_(intermediate_output, attention_output)
        output = layer_output
        return output
    
class TFBertAttention(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.self = TFBertSelfAttention(config, name = "self")
        self.output_ = TFSelfOutput(config, name = "output")

    def call(self, hidden_states):
        self_outputs = self.self(hidden_states)
        attention_output = self.output_(self_outputs, hidden_states)
        outputs = attention_output
        return outputs
    
class TFBertSelfAttention(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.batch_size = 1
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = Linear(self.all_head_size, name = "query")
        self.key = Linear(self.all_head_size, name = "key")
        self.value = Linear(self.all_head_size, name = "value")
        self.softmax = K.layers.Softmax()
        
#         self.dropout = K.layers.Dropout(config.attention_probs_dropout_prob, name = "dropout")
        
    def transpose_for_scores(self, x):
        x = tf.reshape(x, (self.batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return x # tf.transpose(x, (0, 2, 1, 3))
        
    def call(self, hidden_states):
        key_layer = tf.transpose(self.transpose_for_scores(self.key(hidden_states)), (0, 2, 3, 1))
        value_layer = tf.transpose(self.transpose_for_scores(self.value(hidden_states)), (0, 2, 1, 3))
        query_layer = tf.transpose(self.transpose_for_scores(self.query(hidden_states)), (0, 2, 1, 3))
        
        attention_scores = tf.matmul(query_layer, key_layer)
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_probs = self.softmax(attention_scores)
#         attention_probs = self.dropout(attention_probs)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
        context_layer = tf.reshape(context_layer, (self.batch_size, -1, self.all_head_size))
        outputs = context_layer
        return outputs
    
class TFSelfOutput(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.dense = Linear(config.hidden_size, name = "dense")
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
#         self.dropout = K.layers.Dropout(config.hidden_dropout_prob, name = "dropout")

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class TFIntermediate(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.dense = Linear(config.intermediate_size, name = "dense")
        if config.hidden_act == "silu":
            self.intermediate_act_fn = tf.nn.silu

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class TFOutput(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.dense = Linear(config.hidden_size, name = "dense")
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
#         self.dropout = K.layers.Dropout(config.hidden_dropout_prob, name = "dropout")

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class TFBertModel(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.embeddings = TFBertEmbedding(config, name = "embeddings")
        self.encoder = TFBertEncoderCPE(config, name = "encoder")
        
    def call(self, inputs_embeds):
        inputs_embeds = self.embeddings(inputs_embeds)
        output = self.encoder(inputs_embeds)
        return output
    
class ClsEmb(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.hidden_size = config.hidden_size
    def build(self, input_shape):
        self.cls_emb = self.add_weight(name = "cls_emb", shape = (1, 1, self.hidden_size))
    def call(self, inputs_embeds):
        return tf.concat([self.cls_emb, inputs_embeds], 1)
    
class Linear(K.layers.Layer):
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

class OneBertModel(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.hidden_size = config.hidden_size
        self.emb = Linear(config.hidden_size, use_bias = False, name = "emb")
        self.cls_emb = ClsEmb(config, name = "cls_emb")
        self.model = TFBertModel(config, name = "model")
        self.fc = Linear(config.num_labels, name = "fc.1")
        self.softmax = K.layers.Softmax(name = "softmax")
    def build(self, input_shape):
        self.cls_emb = self.add_weight(name = "cls_emb", shape = (1, 1, self.hidden_size))
    def call(self, inputs_embeds):
        inputs_embeds = self.emb(inputs_embeds)
        inputs_embeds = tf.concat([self.cls_emb, inputs_embeds], 1)
        out = self.model(inputs_embeds)[:,0]
        out = self.fc(out)
        out = self.softmax(out[0])
        return out
    
class Div(K.layers.Layer):
    def __init__(self, n, name = None):
        super().__init__(name = name)
        self.n = n
    def call(self, x):
        return x / self.n