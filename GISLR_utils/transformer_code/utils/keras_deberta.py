import tensorflow as tf
import tensorflow.experimental.numpy as tfnp
import keras as K
from .augmentations import *
from .keras_bert import Preprocess, Linear, Div, ClsEmb, TFIntermediate, TFOutput, TFSelfOutput


dis_idx0, dis_idx1 = np.where(np.triu(np.ones((21, 21)), 1) == 1)
dis_idx2, dis_idx3 = np.where(np.triu(np.ones((20, 20)), 1) == 1)


class ConvLayer(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = K.layers.Conv1D(config.hidden_size, kernel_size, padding = "same", groups = groups, name = "conv")
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
        # self.dropout = StableDropout(config.hidden_dropout_prob)
        if self.conv_act == "tanh":
            self.conv_act = tf.tanh
        self.config = config

    def forward(self, hidden_states, residual_states):
        out = self.conv(hidden_states)
        # out = self.dropout(out)
        out = self.conv_act(out)
        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)
        return output


class TFDebertaV2Embedding(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.position_embeddings = K.layers.Embedding(config.max_position_embeddings, config.hidden_size, name = "position_embeddings")
        # self.token_type_embeddings = K.layers.Embedding(2, config.hidden_size, name = "token_type_embeddings")
        self.position_ids = tf.range(config.max_position_embeddings)
        # self.token_type_ids = tf.zeros(config.max_position_embeddings)
        self.LayerNorm = K.layers.LayerNormalization(epsilon = config.layer_norm_eps, name = "LayerNorm")
#         self.dropout = K.layers.Dropout(config.hidden_dropout_prob)
        
    def call(self, inputs_embeds):
        position_ids = self.position_ids[:tf.shape(inputs_embeds)[1]]
        # token_type_ids = self.token_type_ids[:tf.shape(inputs_embeds)[1]]
        embeddings = inputs_embeds 
        embeddings += self.position_embeddings(position_ids)
        # embeddings += self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
        return embeddings

class TFDebertaV2EncoderCPE(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.layer = [TFDebertaV2LayerCPE(config, _, name = f"layer.{_}") for _ in range(config.num_hidden_layers)]
        
    def call(self, hidden_states):
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states
    
class TFDebertaV2LayerCPE(K.layers.Layer):
    def __init__(self, config, layer_idx, name = None):
        super().__init__(name = name)
        self.attention = TFDebertaV2Attention(config, name = "attention")
        self.intermediate = TFIntermediate(config, name = "intermediate")
        self.output_ = TFOutput(config, name = "output")
        
        if config.cpe_start <= layer_idx < config.cpe_end:
            kernel_size = config.cpe_kernel_size
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
    
class TFDebertaV2Attention(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.self = DisentangledSelfAttention(config, name = "self")
        self.output_ = TFSelfOutput(config, name = "output")

    def call(self, hidden_states):
        self_outputs = self.self(hidden_states)
        attention_output = self.output_(self_outputs, hidden_states)
        outputs = attention_output
        return outputs
    
class DisentangledSelfAttention(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.batch_size = 1
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query_proj = Linear(self.all_head_size, name = "query_proj")
        self.key_proj = Linear(self.all_head_size, name = "key_proj")
        self.value_proj = Linear(self.all_head_size, name = "value_proj")
        self.softmax = K.layers.Softmax()
        
#         self.dropout = StableDropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        x = tf.reshape(x, (self.batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (self.batch_size * self.num_attention_heads, -1, self.attention_head_size))
        
    def call(self, hidden_states):
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))
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
    
class TFDebertaV2Model(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.embeddings = TFDebertaV2Embedding(config, name = "embeddings")
        self.encoder = TFDebertaV2EncoderCPE(config, name = "encoder")
        
    def call(self, inputs_embeds):
        inputs_embeds = self.embeddings(inputs_embeds)
        output = self.encoder(inputs_embeds)
        return output
    
class OneDebertaV2Model(K.layers.Layer):
    def __init__(self, config, name = None):
        super().__init__(name = name)
        self.hidden_size = config.hidden_size
        self.emb = Linear(config.hidden_size, use_bias = False, name = "emb")
        self.cls_emb = ClsEmb(config, name = "cls_emb")
        self.model = TFDebertaV2Model(config, name = "model")
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

if __name__ == "__main__":
    import transformers

    base_model_path = "./weights/test/test" + ".ckpt"
    tflite_model_path = base_model_path.replace(".ckpt", ".tflite").replace("=", "-")

    config = transformers.DebertaV2Config()
    config.hidden_size = 256
    config.intermediate_size = 512 # config.hidden_size // 2
    config.num_attention_heads = 4
    config.max_position_embeddings = 96 + 1
    config.num_hidden_layers = 4
    config.vocab_size = 1
    config.pre_emb_size = 972
    config.num_labels = 250
    config.output_hidden_states = True
    config.hidden_act = "silu"
    config.cpe_kernel_size = 3
    config.cpe_start = 1
    config.cpe_end = 4

    input_layer = K.Input((543, 3), name = "inputs")
    inputs_embeds = Preprocess()(input_layer)
    output_layer = OneDebertaV2Model(config, name = "outputs")(inputs_embeds)
    keras_model = K.models.Model(input_layer, output_layer)

    pos = np.random.rand((1, 543, 3))

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    interpreter = tf.lite.Interpreter(tflite_model_path)
    found_signatures = list(interpreter.get_signature_list().keys())
    prediction_fn = interpreter.get_signature_runner("serving_default")
    tf_result = prediction_fn(inputs = pos)
    tf_result = list(tf_result.values())[0]#["outputs"]
    print(tf_result, tf_result.shape)