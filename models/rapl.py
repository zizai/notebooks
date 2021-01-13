import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import (Categorical, Normal)

from constants import UNK_TOKEN_ID
from .controller import Controller


class CNNEncoder(nn.Module):
    """Maps (k, v) sequences to internal representations
    """
    def __init__(self, device, x_dim, h_dim, s_dim):
        super(CNNEncoder, self).__init__()

        self.device = device
        self.h_dim = h_dim

        self.conv1 = nn.Conv1d(h_dim, h_dim, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(h_dim, h_dim * 2, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(h_dim * 2, h_dim, kernel_size=5, stride=1)
        self.mu = nn.Linear(h_dim, s_dim)
        self.var = nn.Linear(h_dim, s_dim)

    def forward(self, k, v, v_len):
        """

        Parameters
        ----------
        k : torch.Tensor
        v : torch.Tensor
            Shape (batch_size, seq_len, h_dim)
        v_len : torch.Tensor
            Shape (batch_size, )

        """
        # reshape to (batch_size, h_dim, seq_len)
        v = v.transpose(1, 2)

        print(v.shape)
        h = F.relu(self.conv1(v))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        print(h.shape)
        h_avg = torch.mean(h, dim=-1)

        mu = self.mu(h_avg)
        var = 1e-6 + F.softplus(self.var(h_avg))

        return h, mu, var


class RNNEncoder(nn.Module):
    """
    Maps a state s to mu and sigma which will define the generative model
    from which we sample new states.
    """
    def __init__(self, device, h_dim, s_dim, n_layers=2, bidirectional=True):
        """
        Parameters
        ----------
        h_dim : int
            Dimension of hidden layer.
        s_dim : int
            Dimension of state representation s.
        """
        super(RNNEncoder, self).__init__()

        self.device = device
        self.h_dim = h_dim
        self.bidirectional = bidirectional

        self.gru = nn.GRU(h_dim, h_dim, n_layers, bidirectional=bidirectional)
        self.mu = nn.Linear(h_dim * n_layers * 2, s_dim)
        self.var = nn.Linear(h_dim * n_layers * 2, s_dim)

    def forward(self, k, v, v_len):
        v = pack_padded_sequence(v, v_len, batch_first=True, enforce_sorted=False)

        # Pass through GRU, obtain final output, and reshape to (B x 2 * l_dim * h_dim)
        _, hidden = self.gru(v, None)
        hidden = hidden.transpose(0, 1).contiguous().view([v_len.shape[0], -1])

        mu = self.mu(hidden)
        var = self.var(hidden)
        var = 1e-6 + F.softplus(var)

        return hidden, mu, var


class CNNDecoder(nn.Module):
    def __init__(self, device, s_dim, h_dim):
        super(CNNDecoder, self).__init__()
        self.device = device

        self.fc = nn.Linear(s_dim, h_dim)
        self.deconv1 = nn.ConvTranspose1d(h_dim, h_dim * 2, kernel_size=5, stride=1)
        self.deconv2 = nn.ConvTranspose1d(h_dim * 2, h_dim, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose1d(h_dim, h_dim, kernel_size=5, stride=2)

    def forward(self, h, s, k, v_mask, v_len):
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))

        # reshape to (batch_size, seq_len, h_dim)
        h = h.transpose(1, 2)
        print(h.shape)
        return h


class RNNDecoder(nn.Module):
    """
    A basic decoder that reconstructs values given keys and internal state
    """
    def __init__(self, device, s_dim, h_dim, n_layers=2, word_dropout=1.):
        super(RNNDecoder, self).__init__()

        self.device = device
        self.s_dim = s_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.word_dropout = word_dropout

        self.gru = nn.GRU(h_dim, h_dim, n_layers)
        self.s2h = nn.Linear(s_dim, h_dim)

    def forward(self, h, s, k, v_mask, v_len, temperature=1.):
        v_mask = pack_padded_sequence(v_mask, v_len, batch_first=True, enforce_sorted=False)
        hidden = self.s2h(s).unsqueeze(0).repeat(self.n_layers, 1, 1)

        h, _ = self.gru(v_mask, hidden)

        max_len = v_len.max().item()
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0, total_length=max_len)
        return h


class RAPL(nn.Module):
    """Recurrent Action Perception Loop
    """
    def __init__(self, device, vocab_size, x_dim, s_dim, a_dim, h_dim=512, encoder_type='RNN', decoder_type='RNN'):
        """
        Parameters
        ----------
        vocab_size : int
            Size of vocabulary.
        x_dim : int
            Dimension of viewpoint
        s_dim : int
            Dimension of state representation s.
        a_dim : int
            Dimension of action a.
        h_dim : int
            Dimension of hidden layer.
        """
        super(RAPL, self).__init__()
        self.device = device

        self.embed = nn.Embedding(vocab_size, h_dim, padding_idx=0)
        self.out = nn.Linear(h_dim, vocab_size)

        if encoder_type == 'CNN':
            self.encoder = CNNEncoder(device, x_dim, h_dim, s_dim)
        elif encoder_type == 'RNN':
            self.encoder = RNNEncoder(device, h_dim, s_dim)
        else:
            raise ValueError(
                "Encoder type not recognized: {}. Please choose [CNN, RNN]".format(encoder_type))

        if decoder_type == 'CNN':
            self.decoder = CNNDecoder(device, s_dim, h_dim)
        elif decoder_type == 'RNN':
            self.decoder = RNNDecoder(device, s_dim, h_dim)
        else:
            raise ValueError(
                "Decoder type not recognized: {}. Please choose [CNN, RNN]".format(encoder_type))

        self.control = Controller(device, s_dim, a_dim)

        # Tying weights in the input and output layer might increase performance (Inan et al., 2016)
        self.out.weight = self.embed.weight

    def forward(self, k_t, v_t, v_len, teacher_forcing=False):
        v_embedded = self.embed(v_t.detach())
        h_t, mu_s, var_s = self.encoder(k_t, v_embedded, v_len)
        q_s = Normal(mu_s, var_s)

        # sample posterior
        s_t = q_s.rsample() # FIXME gradients vanish?

        batch_size = v_len.shape[0]
        max_len = v_len.max().item()
        v_mask = torch.ones((batch_size, max_len), device=self.device, dtype=torch.long) * UNK_TOKEN_ID
        v_mask = self.embed(v_mask)
        h_t = self.decoder(h_t, s_t, k_t, v_mask, v_len)
        v_t_pred = self.out(h_t)

        a_tp1 = self.control(s_t)

        return s_t, a_tp1, q_s, v_t_pred
