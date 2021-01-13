import math
import torch
import torch.nn as nn

from models import sequence_mask
from .embedding import PositionalEncoding
from .transformer_layers import TransformerEncoderLayer, TransformerInterLayer, TransformerDecoderLayer


class TransformerInterEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):
        super(TransformerInterEncoder, self).__init__()
        inter_layers = [int(i) for i in inter_layers]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_blocks, n_tokens = src.size()
        # src = src.view(batch_size * n_blocks, n_tokens)
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif (self.transformer_types[i] == 'inter'):
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim

        word_vec = self.layer_norm(word_vec)
        mask_hier = mask_local[:, :, None].float()
        src_features = word_vec * mask_hier
        src_features = src_features.view(batch_size, n_blocks * n_tokens, -1)
        src_features = src_features.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        mask_hier = mask_hier.view(batch_size, n_blocks * n_tokens, -1)
        mask_hier = mask_hier.transpose(0, 1).contiguous()

        unpadded = [torch.masked_select(src_features[:, i], mask_hier[:, i].byte()).view([-1, src_features.size(-1)])
                    for i in range(src_features.size(1))]
        max_l = max([p.size(0) for p in unpadded])
        mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).to(self.device)
        mask_hier = 1 - mask_hier[:, None, :]

        unpadded = torch.stack(
            [torch.cat([p, torch.zeros(max_l - p.size(0), src_features.size(-1)).to(self.device)]) for p in unpadded], 1)
        return unpadded, mask_hier


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_hier = 1 - src.data.eq(padding_idx)
        out = self.pos_emb(emb)

        for i in range(self.num_layers):
            out = self.transformer_local[i](out, out, 1 - mask_hier)  # all_sents * max_tokens * dim
        out = self.layer_norm(out)

        mask_hier = mask_hier[:, :, None].float()
        src_features = out * mask_hier
        src_features = src_features.transpose(0, 1).contiguous()
        mask_hier = mask_hier.transpose(0, 1).contiguous()
        # bridge_feature = self._bridge(src_features, mask_hier)

        # return bridge_feature, src_features, mask_hier


MAX_SIZE = 5000


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """

        src = state.src
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        output = self.pos_emb(output, step)

        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)


        for i in range(self.num_layers):
            output, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()


        return outputs, state

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        if(src.dim()==3):
            src = src.view(src.size(0),-1).transpose(0,1)
        else:
            src = src.transpose(0,1)
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {"memory_keys": None, "memory_values": None, "self_keys": None, "self_values": None}
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 1)
        if self.cache is not None:
            _recursive_map(self.cache)