import math
import numpy as np
from scipy.stats import poisson
import scipy.sparse as sps

from models.sparse import SparseLinear, SparseGCN
from utils import scatter_add, coalesce
from .module import Module, ModuleList
from .parameter import Parameter


def edge_softmax(src, index, num_nodes):
    src = np.exp(src)
    norm = np.zeros(num_nodes)
    norm = scatter_add(norm, index, src)
    out = src / (norm[index] + 1e-16)
    return out


def relu(x): return np.maximum(x, 0)


def sigmoid(x):
    return (1 + np.exp(-x)) ** -1


class NodeAssembly(Module):
    def __init__(self, in_channels, dropout=0.):
        super(NodeAssembly, self).__init__()
        self.dropout = dropout
        self.xx_to_score = SparseLinear(in_channels * 2, 1, bias=False)

    def forward(self, x, edge_index):
        if edge_index.shape[1] > 0:
            num_nodes = x.shape[0]
            # compute edge scores from incident nodes
            xx = np.concatenate([x[edge_index[0]], x[edge_index[1]]], axis=-1)
            edge_score = self.xx_to_score.forward(xx)
            edge_score = edge_score.reshape(-1)

            # softmax over edge scores with regard to each node
            edge_score = edge_softmax(edge_score, edge_index[1], num_nodes)

            x, edge_index = self.__merge_edges__(x, edge_index, edge_score)
        return x, edge_index

    def __merge_edges__(self, x, edge_index, edge_score):
        nodes_remaining = set(range(x.shape[0]))

        cluster = np.zeros(x.shape[0], dtype=edge_index.dtype)
        # argsort in ascending order and reverse
        edge_argsort = np.argsort(edge_score)[::-1]

        # Iterate through all edges
        i = 0
        new_edge_indices = []
        for edge_idx in edge_argsort.tolist():
            source = edge_index[0, edge_idx]
            target = edge_index[1, edge_idx]

            if source in nodes_remaining and target in nodes_remaining:
                # contract the edge if it is not incident to a chosen node
                new_edge_indices.append(edge_idx)

                cluster[source] = i
                nodes_remaining.remove(source)

                if source != target:
                    cluster[target] = i
                    nodes_remaining.remove(target)

                i += 1
            else:
                continue

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1

        # We compute the new features as an addition of the old ones.
        new_num_nodes = np.max(cluster) + 1
        new_x = np.zeros((new_num_nodes, x.shape[1]), dtype=x.dtype)
        new_x = scatter_add(new_x, cluster, x)

        N = new_x.shape[0]
        new_edge_index = coalesce(cluster[edge_index], None, N, N)
        new_edge_index = np.array(new_edge_index.nonzero(), dtype=edge_index.dtype)

        return new_x, new_edge_index


class NodeReplication(Module):
    def __init__(self, in_channels, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.x_to_score = SparseLinear(in_channels, 1)
        self.k_net = SparseLinear(in_channels, in_channels)
        self.q_net = SparseLinear(in_channels, in_channels)
        self.att_to_kai = Parameter(sps.rand(in_channels, 1, density=0.5))
        self.att_to_theta = Parameter(sps.rand(in_channels, 1, density=0.5))
        self.temp = lambda n: 1e+100 ** (1 / n)

    def forward(self, x, edge_index):
        # mask nodes going to split
        temperature = self.temp(x.shape[0])
        scores = self.x_to_score.forward(x) * temperature
        logits = sigmoid(scores).reshape(-1)
        x_mask = logits > 0

        if x_mask.sum(0) <= 0:
            return x, edge_index
        else:
            # split nodes
            new_x = x[x_mask, :] / 2
            x[x_mask, :] = new_x

            num_old_nodes = x.shape[0]
            num_new_nodes = new_x.shape[0]
            x = np.concatenate([x, new_x], axis=0)

            # make new edges
            # compute attention scores between new nodes and all nodes
            query = self.q_net.forward(new_x)
            key = self.k_net.forward(x)
            att_scores = np.einsum('in,jn->ijn', query, key)
            assert att_scores.shape[0] == num_new_nodes

            # compute parameters for bernoulli mixture
            kai = relu(att_scores @ self.att_to_kai.toarray()).sum(axis=1).reshape(-1)
            theta = sigmoid(att_scores @ self.att_to_theta.toarray() * temperature).reshape(num_new_nodes, -1)

            assert kai.shape[0] == theta.shape[0]
            if kai.shape[0] <= 0:
                return x, edge_index
            else:
                edge_index = [edge_index]
                for i, logits in enumerate(theta):
                    # sample k neighbors
                    k = kai[i]
                    k = poisson.rvs(k, 1)
                    if k == 0:
                        continue
                    k_neighbors = np.argsort(logits)[:-k]

                    if k_neighbors.shape[0] == 0:
                        continue

                    # add new edges
                    source = i + num_old_nodes
                    new_edges = [np.ones_like(k_neighbors) * source, k_neighbors]
                    new_edges = np.stack(new_edges)
                    edge_index.append(new_edges)

                edge_index = np.concatenate(edge_index, axis=-1)
                return x, edge_index


class GraphEncoderLayer(Module):
    def __init__(self, r_dim, density=0.01, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.conv = SparseGCN(r_dim, r_dim, density=density)
        self.pool = NodeAssembly(r_dim)

    def forward(self, x, edge_index):
        r = self.conv.forward(x, edge_index)
        r, edge_index = self.pool.forward(r, edge_index)
        return r, edge_index


class GraphEncoder(Module):
    def __init__(self, r_dim, num_layers=3, density=0.01, dropout=0.):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        layer = GraphEncoderLayer(r_dim, density=density, dropout=dropout)
        self.layers = ModuleList([layer] * self.num_layers)

    def forward(self, x, edge_index):
        for _, layer in enumerate(self.layers):
            x, edge_index = layer.forward(x, edge_index)
        return x, edge_index


class GraphDecoderLayer(Module):
    def __init__(self, r_dim, density=0.01, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.conv = SparseGCN(r_dim, r_dim, density=density)
        self.unpool = NodeReplication(r_dim, dropout=self.dropout)

    def forward(self, x, edge_index):
        r = self.conv.forward(x, edge_index)
        r, edge_index = self.unpool.forward(r, edge_index)
        return r, edge_index


class GraphDecoder(Module):
    def __init__(self, r_dim, num_layers=3, density=0.01, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        layer = GraphDecoderLayer(r_dim, density=density, dropout=dropout)
        self.layers = ModuleList([layer] * self.num_layers)

    def forward(self, x, edge_index):
        for _, layer in enumerate(self.layers):
            x, edge_index = layer.forward(x, edge_index)
        return x, edge_index
