import torch
import torch.nn as nn

from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Attention, DebertaV2Intermediate, DebertaV2Output, LayerNorm, ConvLayer, build_relative_position, Sequence, BaseModelOutput, DebertaV2Encoder


class DebertaV2EncoderCPE(DebertaV2Encoder):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([DebertaV2LayerCPE(config, _) for _ in range(config.num_hidden_layers)])

class DebertaV2LayerCPE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

        if config.cpe_start <= layer_idx < config.cpe_end:
            kernel_size = config.cpe_kernel_size
            self.cpe_conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, 1, padding = kernel_size // 2, groups = config.hidden_size)
        else:
            self.cpe_conv = None

    def cpe(self, hidden_states):
        if self.cpe_conv is not None:
            hidden_states = hidden_states.permute(0, 2, 1)
            hidden_states += self.cpe_conv(hidden_states)
            return hidden_states.permute(0, 2, 1)
        else:
            return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        hidden_states = self.cpe(hidden_states)
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output
