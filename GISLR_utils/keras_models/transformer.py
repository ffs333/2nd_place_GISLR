import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tfnp

import keras as K

from .common import KerasTransformerLinear


class KerasBertEmbedding(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.position_embeddings = K.layers.Embedding(config.max_position_embeddings, config.hidden_size, name = "position_embeddings")
        self.position_ids = tf.range(config.max_position_embeddings)
        if config.model_type != "deberta-v2":
            self.token_type_embeddings = K.layers.Embedding(2, config.hidden_size, name = "token_type_embeddings")
            self.token_type_ids = tf.zeros(config.max_position_embeddings)
        else:
            self.token_type_embeddings = None
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
#         self.dropout = K.layers.Dropout(config.hidden_dropout_prob)
        
    def call(self, inputs_embeds):
        embeddings = inputs_embeds 
        position_ids = self.position_ids[:tf.shape(inputs_embeds)[1]]
        embeddings += self.position_embeddings(position_ids)
        if self.token_type_embeddings is not None:
            token_type_ids = self.token_type_ids[:tf.shape(inputs_embeds)[1]]
            embeddings += self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
        return embeddings

class KerasBertEncoderCPE(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.layer = [KerasBertLayerCPE(config, _, name = f"layer.{_}") for _ in range(config.num_hidden_layers)]
        
    def call(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)
        return hidden_states
    
class KerasBertLayerCPE(K.layers.Layer):
    def __init__(self, config, layer_idx, name = None):
        super().__init__(name = name)
        is_last = layer_idx == config.num_hidden_layers - 1
        self.attention = KerasBertAttention(config, is_last = is_last, name = "attention")
        self.intermediate = KerasIntermediate(config, name = "intermediate")
        self.output_ = KerasOutput(config, name = "output")
        
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
    
class KerasBertAttention(K.layers.Layer):
    def __init__(self, config, is_last = False, name = None):
        super().__init__(name = name)
        if config.model_type == "deberta-v2":
            self.self = KerasDisentangledSelfAttention(config, is_last = is_last, name = "self")
        else:
            self.self = KerasBertSelfAttention(config, is_last = is_last, name = "self")
        self.output_ = KerasSelfOutput(config, is_last = is_last, name = "output")

    def call(self, hidden_states):
        self_outputs = self.self(hidden_states)
        attention_output = self.output_(self_outputs, hidden_states)
        outputs = attention_output
        return outputs
    
class KerasBertSelfAttention(K.layers.Layer):
    def __init__(self, config, is_last = False, name = None):
        super().__init__(name = name)
        self.is_last = is_last
        self.batch_size = 1
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = KerasTransformerLinear(self.all_head_size, name = "query")
        self.key = KerasTransformerLinear(self.all_head_size, name = "key")
        self.value = KerasTransformerLinear(self.all_head_size, name = "value")
        self.softmax = K.layers.Softmax()
        
#         self.dropout = K.layers.Dropout(config.attention_probs_dropout_prob, name = "dropout")
        
    def transpose_for_scores(self, x):
        x = tf.reshape(x, (self.batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return x # tf.transpose(x, (0, 2, 1, 3))
        
    def call(self, hidden_states):
        key_layer = tf.transpose(self.transpose_for_scores(self.key(hidden_states)), (0, 2, 3, 1))
        value_layer = tf.transpose(self.transpose_for_scores(self.value(hidden_states)), (0, 2, 1, 3))
        if self.is_last:
            query_layer = tf.transpose(self.transpose_for_scores(self.query(hidden_states[:,0:1])), (0, 2, 1, 3))
        else:
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

class KerasDisentangledSelfAttention(K.layers.Layer):
    def __init__(self, config, is_last = False, name = None):
        super().__init__(name = name)
        self.is_last = is_last
        self.batch_size = 1
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query_proj = KerasTransformerLinear(self.all_head_size, name = "query_proj")
        self.key_proj = KerasTransformerLinear(self.all_head_size, name = "key_proj")
        self.value_proj = KerasTransformerLinear(self.all_head_size, name = "value_proj")
        self.softmax = K.layers.Softmax()
        
#         self.dropout = StableDropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        x = tf.reshape(x, (self.batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (self.batch_size * self.num_attention_heads, -1, self.attention_head_size))
        
    def call(self, hidden_states):
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))
        if self.is_last:
            query_layer = self.transpose_for_scores(self.query_proj(hidden_states[:,0:1]))
        else:
            query_layer = self.transpose_for_scores(self.query_proj(hidden_states))
        
        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, (0, 2, 1)))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        L = tf.shape(attention_scores)[-1]
        attention_scores = tf.reshape(attention_scores, (self.batch_size, self.num_attention_heads, -1, L))
        attention_probs = self.softmax(attention_scores)
#         attention_probs = self.dropout(attention_probs)

        context_layer = tf.matmul(
            tf.reshape(attention_probs, (self.batch_size * self.num_attention_heads, -1, L)), value_layer)
        context_layer = tf.reshape(context_layer, (self.batch_size, self.num_attention_heads, -1, self.attention_head_size))
        context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
        context_layer = tf.reshape(context_layer, (self.batch_size, -1, self.all_head_size))
        outputs = context_layer
        return outputs
    
class KerasSelfOutput(K.layers.Layer):
    def __init__(self, config, is_last = False, name = None):
        super().__init__(name = name)
        self.is_last = is_last
        self.dense = KerasTransformerLinear(config.hidden_size, name = "dense")
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
#         self.dropout = K.layers.Dropout(config.hidden_dropout_prob, name = "dropout")

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
        if self.is_last:
            hidden_states = self.LayerNorm(hidden_states + input_tensor[:,0:1])
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class KerasIntermediate(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.dense = KerasTransformerLinear(config.intermediate_size, name = "dense")
        if config.hidden_act == "silu":
            self.intermediate_act_fn = tf.nn.silu

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class KerasOutput(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.dense = KerasTransformerLinear(config.hidden_size, name = "dense")
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
#         self.dropout = K.layers.Dropout(config.hidden_dropout_prob, name = "dropout")

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class KerasBertModel(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.embeddings = KerasBertEmbedding(config, name = "embeddings")
        self.encoder = KerasBertEncoderCPE(config, name = "encoder")
        
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

class KerasTransformer(K.layers.Layer):
    def __init__(self, config, softmax = True, name = None):
        super().__init__(name = name)
        self.hidden_size = config.hidden_size
        self.emb = KerasTransformerLinear(config.hidden_size, use_bias = False, name = "emb")
        self.cls_emb = ClsEmb(config, name = "cls_emb")
        self.model = KerasBertModel(config, name = "model")
        self.fc = KerasTransformerLinear(config.num_labels, name = "fc.1")
        self.softmax = K.layers.Softmax(name = "softmax")
        if not softmax:
            self.softmax = None
    def build(self, input_shape):
        self.cls_emb = self.add_weight(name = "cls_emb", shape = (1, 1, self.hidden_size))
    def call(self, inputs_embeds):
        inputs_embeds = self.emb(inputs_embeds)
        inputs_embeds = tf.concat([self.cls_emb, inputs_embeds], 1)
        out = self.model(inputs_embeds)[:,0]
        out = self.fc(out)
        if self.softmax is not None:
            out = self.softmax(out[0])
        else:
            out = out[0]
        return out