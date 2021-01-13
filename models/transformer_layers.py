import sys
import math
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (MultiHeadedAttention, MultiHeadedPooling)

from collections import defaultdict

MAX_SIZE = 5000


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        context = self.pooling_attn(inputs, inputs,
                                    mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)


class TransformerInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerInterLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.inter_att = MultiHeadedAttention(1, self.d_per_head, dropout, use_final_linear=False)

        self.linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask_local, mask_inter, batch_size, n_blocks):
        word_vec = self.layer_norm1(inputs)
        mask_inter = mask_inter.unsqueeze(1).expand(batch_size, self.heads, n_blocks).contiguous()
        mask_inter = mask_inter.view(batch_size * self.heads, 1, n_blocks)

        # block_vec = self.pooling(word_vec, mask_local)

        block_vec = self.pooling(word_vec, word_vec, mask_local)
        block_vec = block_vec.view(-1, self.d_per_head)
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, self.heads, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * self.heads, n_blocks, self.d_per_head)

        block_vec = self.inter_att(block_vec, block_vec, block_vec, mask_inter)  # all_sents * max_tokens * dim
        block_vec = block_vec.view(batch_size, self.heads, n_blocks, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * n_blocks, self.heads * self.d_per_head)
        block_vec = self.linear(block_vec)

        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out


class TransformerNewInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerNewInterLayer, self).__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout)

        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.inter_att = MultiHeadedAttention(heads, d_model, dropout, use_final_linear=True)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask_local, mask_inter, batch_size, n_blocks):
        word_vec = self.layer_norm1(inputs)
        mask_inter = mask_inter.unsqueeze(1)
        # block_vec = self.pooling(word_vec, mask_local)

        block_vec = self.pooling(word_vec, word_vec, mask_local)
        _mask_local = ((1 - mask_local).unsqueeze(-1)).float()
        block_vec_avg = torch.sum(word_vec * _mask_local, 1) / (torch.sum(_mask_local, 1) + 1e-9)
        block_vec = self.dropout(block_vec) + block_vec_avg
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, -1)
        block_vec = self.inter_att(block_vec, block_vec, block_vec, mask_inter)  # all_sents * max_tokens * dim
        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`
        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):
            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`
        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm

        query = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, all_input
        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.
        Args:
            size: int
        Returns:
            (`LongTensor`):
            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask
