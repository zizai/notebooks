import itertools
import math
from collections import OrderedDict

import networkx as nx
import numpy as np
from scipy import stats
import scipy.sparse as sps

from utils import coalesce
from .module import Module
from .parameter import Parameter


class SparseEmbedding(Module):
    def __init__(self, embedding_dim):
        super(SparseEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: np.array):
        assert x.dtype == np.int
        x_len = x.shape[0]
        output = sps.dok_matrix((x_len, self.embedding_dim))

        for i, val in enumerate(x):
            output[i, val] = 1

        return output


class SparseGCN(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 density=0.1):
        super(SparseGCN, self).__init__()
        self.density = density
        self.w = Parameter(sps.coo_matrix((in_channels, out_channels)))

        self.reset_parameters()

    def reset_parameters(self):
        self.w.glorot_uniform(self.density)

    def forward(self, x: np.ndarray, edge_index: np.ndarray):
        """

        Parameters
        ----------
        x
        edge_index

        Returns
        -------

        """
        x_len, x_dim = x.shape
        z = sps.csr_matrix(x)
        z = z @ self.w

        # make sparse adjacency matrix
        adj = coalesce(edge_index, None, x_len, x_len).tocsr()

        # add self loops
        idx = range(x_len)
        adj[idx, idx] = 2

        # compute message weights
        deg_inv_sqrt = sps.diags(np.array(adj.sum(axis=-1)).clip(1).reshape(-1) ** -0.5)
        A = deg_inv_sqrt @ adj @ deg_inv_sqrt

        # do message passing
        z = A @ z

        if type(x) is np.ndarray:
            z = z.toarray()
        return z


class SparseLinear(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 density=0.1,
                 bias=False):
        super(SparseLinear, self).__init__()
        self.density = density
        self.weight = Parameter(sps.coo_matrix((in_channels, out_channels)))
        self.bias = Parameter(sps.coo_matrix((1, out_channels))) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.glorot_uniform(self.density)
        if self.bias is not None:
            self.bias.glorot_uniform(self.density)

    def forward(self, x: np.ndarray):
        x_len, x_dim = x.shape

        z = sps.csr_matrix(x)
        z = z @ self.weight
        if self.bias is not None:
            bias = self.bias.tocsr()
            out = [z[i] + bias for i in range(x_len)]
            z = sps.vstack(out)

        if type(x) is np.ndarray:
            z = z.toarray()
        return z
