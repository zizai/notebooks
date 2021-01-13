import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson

from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import softmax


class NodeAssembly(nn.Module):
    def __init__(self, in_channels, dropout=0., pooling_op='mean'):
        super(NodeAssembly, self).__init__()
        self.dropout = dropout
        assert pooling_op in ['add', 'max', 'mean']
        self.pooling_op = pooling_op
        self.xx_to_score = nn.Linear(2 * in_channels, 1)

    def forward(self, x, edge_index):
        if edge_index.size(1) > 0:
            num_nodes = x.size(0)
            xx = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
            e = self.xx_to_score(xx).view(-1)
            e = F.dropout(e, p=self.dropout, training=self.training)
            e = softmax(e, edge_index[1], num_nodes)

            x, edge_index = self.__merge_edges__(x, edge_index, e)
        return x, edge_index

    def __merge_edges__(self, x, edge_index, edge_score):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1

        # compute the new features
        if self.pooling_op == 'mean':
            new_x = scatter_mean(x, cluster, dim=0, dim_size=i)
        elif self.pooling_op == 'max':
            new_x = scatter_max(x, cluster, dim=0, dim_size=i)
        elif self.pooling_op == 'add':
            new_x = scatter_add(x, cluster, dim=0, dim_size=i)
            new_x_score = edge_score[new_edge_indices]
            if len(nodes_remaining) > 0:
                remaining_score = x.new_ones(
                    (new_x.size(0) - len(new_edge_indices), ))
                new_x_score = torch.cat([new_x_score, remaining_score])
            new_x = new_x * new_x_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        return new_x, new_edge_index


class NodeReplication(nn.Module):
    def __init__(self, in_channels, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.x_to_score = nn.Linear(in_channels, 1)
        self.k_net = nn.Linear(in_channels, in_channels)
        self.q_net = nn.Linear(in_channels, in_channels)
        self.att_to_kai = nn.Linear(in_channels, 1)
        self.att_to_theta = nn.Linear(in_channels, 1)
        self.temp = lambda n: 1e+100 ** (1 / n)
        self.min_k = 3

    def forward(self, x, edge_index):
        # make new nodes
        # nodes going to split
        temperature = self.temp(x.size(0))
        logits = torch.sigmoid(self.x_to_score(x) * temperature).squeeze(-1)
        logits = F.dropout(logits, self.dropout, self.training)
        x_mask = torch.bernoulli(logits).bool()

        if x_mask.sum(0) <= 0:
            return x, edge_index
        else:
            # split nodes
            # new_x = x[x_mask] / 2
            # x[x_mask] = new_x
            new_x = x[x_mask]

            num_old_nodes = x.size(0)
            num_new_nodes = new_x.size(0)
            x = torch.cat([x, new_x], dim=0)

            # make new edges
            # compute attention scores between new nodes and all nodes
            query = self.q_net(new_x)
            key = self.k_net(x)
            att_scores = torch.einsum('in,jn->ijn', [query, key])

            # compute parameters for bernoulli mixture
            kai = F.relu(self.att_to_kai(att_scores)).sum(1).squeeze(-1)
            theta = torch.sigmoid(self.att_to_theta(att_scores) * temperature).squeeze(-1)

            assert kai.size(0) == theta.size(0)
            if kai.size(0) <= 0:
                return x, edge_index
            else:
                edge_index = [edge_index]
                num_all_nodes = theta.size(1)
                for i, logits in enumerate(theta):
                    k = kai[i]
                    k = Poisson(k).sample().int()

                    neighbors = torch.tensor([])
                    if 0 < k < num_all_nodes:
                        # sample from k nodes
                        logits, indices = torch.topk(logits, k)
                        edge_mask = torch.bernoulli(logits).bool()
                        neighbors = indices[edge_mask]
                    elif k >= num_all_nodes:
                        # sample from all nodes
                        neighbors = torch.bernoulli(logits).nonzero().squeeze()

                    if neighbors.size(0) == 0:
                        continue

                    # add new edges
                    source = i + num_old_nodes
                    new_edges = [torch.ones_like(neighbors) * source, neighbors]
                    new_edges = torch.stack(new_edges)
                    edge_index.append(new_edges)

                edge_index = torch.cat(edge_index, dim=-1)
                return x, edge_index


class GraphEncoderLayer(nn.Module):
    def __init__(self, r_dim, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.conv = GCNConv(r_dim, r_dim, improved=True)
        # self.conv = GATConv(r_dim, r_dim, 4, concat=False)
        self.pool = NodeAssembly(r_dim, dropout=self.dropout)

    def forward(self, x, edge_index):
        r = self.conv(x, edge_index)
        r = F.dropout(F.relu(r), self.dropout, self.training)
        r, edge_index = self.pool(r, edge_index)
        return r, edge_index


class GraphEncoder(nn.Module):
    def __init__(self, r_dim, num_layers=3, dropout=0.):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        layer = GraphEncoderLayer(r_dim, dropout=dropout)
        self.layers = nn.ModuleList([layer] * self.num_layers)

    def forward(self, x, edge_index):
        for _, layer in enumerate(self.layers):
            x, edge_index = layer(x, edge_index)
        return x, edge_index


class GraphDecoderLayer(nn.Module):
    def __init__(self, r_dim, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.conv = GCNConv(r_dim, r_dim, improved=True)
        self.unpool = NodeReplication(r_dim, dropout=self.dropout)

    def forward(self, x, edge_index):
        r = self.conv(x, edge_index)
        # r = F.relu(r)
        r = F.dropout(r, self.dropout, self.training)
        r, edge_index = self.unpool(r, edge_index)
        return r, edge_index


class GraphDecoder(nn.Module):
    def __init__(self, r_dim, num_layers=3, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        layer = GraphDecoderLayer(r_dim, dropout=dropout)
        self.layers = nn.ModuleList([layer] * self.num_layers)

    def forward(self, x, edge_index):
        for _, layer in enumerate(self.layers):
            x, edge_index = layer(x, edge_index)
        return x, edge_index
