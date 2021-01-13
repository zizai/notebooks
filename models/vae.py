import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.utils import dense_to_sparse
from torch.nn import functional as F
from torch.distributions import (Normal)

from utils import seq_to_graph
from .dgn import GraphEncoder, GraphDecoder


'''
https://github.com/Tiiiger/SGC
'''


class VAE(nn.Module):
    def __init__(self, encoder, decoder, z_dim):
        # TODO flow
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = encoder
        self.decoder = decoder
        self.r_to_z = nn.Linear(self.z_dim, 2 * self.z_dim)
        self.prior = Normal(torch.zeros(z_dim), 1)

    def encode(self, *args):
        return self.encoder(*args)

    def decode(self, *args):
        return self.decoder(*args)

    def reparameterize(self, r):
        mu, sigma = torch.chunk(self.r_to_z(r), 2, dim=-1)
        sigma = 1e-6 + F.softplus(sigma)
        p_z = Normal(mu, sigma)
        return p_z

    def sample_prior(self, n):
        z = self.prior.rsample((n,))
        return z


class VGAE(VAE):
    def __init__(self, r_dim, num_encoder_layers=6, num_decoder_layers=6, readout_steps=5, dropout=0.):
        # TODO batch
        # TODO edge attrs
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        encoder = GraphEncoder(r_dim, num_layers=self.num_encoder_layers, dropout=self.dropout)
        decoder = GraphDecoder(r_dim, num_layers=self.num_decoder_layers, dropout=self.dropout)
        super(VGAE, self).__init__(encoder, decoder, r_dim)

        self.set2set = gnn.Set2Set(r_dim, processing_steps=readout_steps)

    def encode(self, x, edge_index=None):
        if edge_index is None:
            # initialize a chain graph
            edge_index = seq_to_graph(x)

        r, edge_index = self.encoder(x, edge_index)
        return r, edge_index

    def decode(self, z, edge_index=None):
        if edge_index is None:
            # inner product decoder
            adj = torch.relu(torch.matmul(z, z.t()))
            # take nonzero elements
            edge_index, _ = dense_to_sparse(adj)

        x, edge_index = self.decoder(z, edge_index)
        return x, edge_index

    def readout(self, r):
        batch = torch.zeros(r.size(0), dtype=torch.long, device=r.device)
        q = self.set2set(r, batch)
        return q
