import time

import networkx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

import torch_geometric.nn as gnn

from torch_cluster import knn

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

ALL_KERNELS = ['cosine', 'l1', 'l2']
ALL_POLICIES = ['1NN']


"""helpers"""


def compute_similarities(query_key, key_list, metric):
    """Compute the similarity between query vs. key_i for all i
        i.e. compute q M, w/ q: 1 x key_dim, M: key_dim x #keys

    Parameters
    ----------
    query_key : a row vector
        Description of parameter `query_key`.
    key_list : list
        Description of parameter `key_list`.
    metric : str
        Description of parameter `metric`.

    Returns
    -------
    a row vector w/ len #memories
        the similarity between query vs. key_i, for all i

    """
    query_key = query_key.data
    # reshape query to 1 x key_dim
    q = query_key.view(1, -1)
    # reshape memory keys to #keys x key_dim
    M = torch.stack(key_list, dim=1).view(len(key_list), -1)
    # compute similarities
    if metric is 'cosine':
        similarities = F.cosine_similarity(q, M)
    elif metric is 'l1':
        similarities = - F.pairwise_distance(q, M, p=1)
    elif metric is 'l2':
        similarities = - F.pairwise_distance(q, M, p=2)
    else:
        raise ValueError(f'unrecog metric: {metric}')
    return similarities


def empty_memory(memory_dim):
    """Get a empty memory, assuming the memory is a row vector
    """
    return torch.zeros(1, memory_dim)


class DND(object):
    """The differentiable neural dictionary (DND) class. This enables episodic
    recall in a neural network.

    Attributes
    ----------
    encoding_off : bool
        if True, stop forming memories
    retrieval_off : type
        if True, stop retrieving memories
    """
    keys = []
    vals = []

    def __init__(self, dict_len, memory_dim, kernel='l2'):
        """
        Parameters
        ----------
        dict_len : int
            the maximial len of the dictionary
        memory_dim : int
            the dim or len of memory i, we assume memory_i is a row vector
        kernel : str
            the metric for memory search
        """
        # params
        self.dict_len = dict_len
        self.kernel = kernel
        self.memory_dim = memory_dim
        # dynamic state
        self.encoding_off = False
        self.retrieval_off = False
        # allocate space for memories
        self.reset_memory()
        # check everything
        self.check_config()

    def reset_memory(self):
        self.keys = []
        self.vals = []

    def check_config(self):
        assert self.dict_len > 0
        assert self.kernel in ALL_KERNELS

    def inject_memories(self, input_keys, input_vals):
        """Inject pre-defined keys and values

        Parameters
        ----------
        input_keys : list
            a list of memory keys
        input_vals : list
            a list of memory content
        """
        assert len(input_keys) == len(input_vals)
        for k, v in zip(input_keys, input_vals):
            self.save_memory(k, v)

    def _get_memory(self, similarities, policy='1NN'):
        """get the episodic memory according to some policy
        e.g. if the policy is 1nn, return the best matching memory
        e.g. the policy can be based on the rational model

        Parameters
        ----------
        similarities : a vector of len #memories
            the similarity between query vs. key_i, for all i
        policy : str
            the retrieval policy

        Returns
        -------
        a row vector
            a DND value, representing the memory content

        """
        best_memory_val = None
        if policy is '1NN':
            best_memory_id = torch.argmax(similarities).item()
            best_memory_val = self.vals[best_memory_id]
        else:
            raise ValueError(f'unrecog recall policy: {policy}')
        return best_memory_val

    def read(self, query_key):
        """Perform a 1-NN search over dnd

        Parameters
        ----------
        query_key : a row vector
            a DND key, used to for memory search

        Returns
        -------
        a row vector
            a DND value, representing the memory content

        """
        # if no memory, return the zero vector
        n_memories = len(self.keys)
        if n_memories == 0 or self.retrieval_off:
            return empty_memory(self.memory_dim)
        # compute similarity(query, memory_i ), for all i
        similarities = compute_similarities(query_key, self.keys, self.kernel)
        # get the best-match memory
        best_memory_val = self._get_memory(similarities)
        return best_memory_val

    def write(self, memory_key, memory_val):
        """Save an episodic memory to the dictionary

        Parameters
        ----------
        memory_key : a row vector
            a DND key, used to for memory search
        memory_val : a row vector
            a DND value, representing the memory content
        """
        if self.encoding_off:
            return
        # add new memory to the the dictionary
        # get data is necessary for gradient reason
        self.keys.append(memory_key.data.view(1, -1))
        self.vals.append(memory_val.data.view(1, -1))
        # remove the oldest memory, if overflow
        if len(self.keys) > self.dict_len:
            self.keys.pop(0)
            self.vals.pop(0)


class KanervaMemory(nn.Module):
    def __init__(self, mem_len, z_dim, k=7, num_heads=8, bias=False):
        super().__init__()
        self.mem_len = mem_len
        self.z_dim = z_dim
        self.k = k
        # graph convolution for global features
        self.gcn = gnn.GCNConv(z_dim, z_dim, improved=True, bias=bias)
        # aggregate local features
        assert z_dim % num_heads == 0
        self.readout = nn.MultiheadAttention(z_dim, num_heads=num_heads, bias=bias)
        self.r_to_y = nn.Linear(self.z_dim, 2 * self.z_dim)

        self.keys = torch.randn((self.mem_len, self.z_dim))
        self.vals = torch.randn_like(self.keys)
        self.register_buffer('keys', self.keys)
        self.register_buffer('values', self.vals)

    @property
    def p_y(self):
        return Normal(torch.zeros(self.z_dim), 1)

    def q_y(self, r):
        mu, sigma = torch.chunk(self.r_to_y(r), 2, dim=-1)
        sigma = 1e-6 + F.softplus(sigma)
        q_y = Normal(mu, sigma)
        return q_y

    def read(self, y):
        """query the memory

        Parameters
        ----------
        y : torch.Tensor
        query vector

        Returns
        -------
        aggregated local features
        """
        source, target = knn(self.keys, y, self.k)
        keys = self.keys[target]
        vals = self.vals[target]
        vals = self.readout(y, keys, vals)
        return vals

    def write(self, r):
        r_len = r.size(0)
        y = self.q_y(r).rsample()
        new_keys = torch.cat([y, self.keys], dim=0)
        new_vals = torch.cat([r, self.vals], dim=0)

        # cluster memory with dynamic graph networks
        edge_index = gnn.knn_graph(new_keys, self.k)
        new_vals = self.gcn(new_vals, edge_index)

        new_vals = new_vals[r_len:]
        new_keys = self.q_y(new_vals).rsample()
        self.keys = new_keys
        self.vals = new_vals


class DTM(nn.Module):
    """Discrete Topological Memory
    """
    def __init__(self, mem_len, z_dim, k=7, bias=False):
        super().__init__()
        self.mem_len = mem_len
        self.z_dim = z_dim
        self.k = k
        self.gcn = gnn.GCNConv(z_dim, z_dim, improved=True, bias=bias)
        self.r_to_y = nn.Linear(self.z_dim, self.mem_len)

        self.mems = torch.randn((self.mem_len, self.z_dim))
        edges = networkx.erdos_renyi_graph(self.mem_len, 10/self.mem_len, ).edges
        src, tgt = torch.tensor([e for e in edges]).t()
        self.edge_index = torch.stack([tgt, src], dim=0)

    @property
    def p_y(self):
        return Categorical(torch.ones((1, self.mem_len)))

    def q_y(self, r):
        weights = F.softplus(self.r_to_y(r))
        q_y = Categorical(weights)
        return q_y

    def to_graph(self):
        x = self.mems
        edge_index = self.edge_index

        data = Data(x=x, edge_index=edge_index)
        graph = to_networkx(data)
        return graph

    def read(self, y: torch.Tensor):
        assert y.dim() == 1
        return self.mems[y]

    def write(self, y: torch.Tensor, r: torch.Tensor):
        # overwrite memory
        self.mems[y] = r

        # remove old edges
        edge_mask = self.edge_index[1] != y
        edge_mask = edge_mask.repeat(2).view(2, -1)
        new_edge_index = self.edge_index.masked_select(edge_mask).view(2, -1)

        # add new edges
        src, tgt = knn(self.mems, r, self.k)
        src = src + y
        y_edge_index = torch.stack([tgt, src], 0)
        new_edge_index = torch.cat([new_edge_index, y_edge_index], dim=-1)

        # propagate changes
        new_mems = self.gcn(self.mems, new_edge_index)

        # reset memory
        self.mems = new_mems
        self.edge_index = new_edge_index

    def shortest_path(self, y: torch.Tensor, y_goal: torch.Tensor):
        graph = self.to_graph()
        try:
            path = networkx.shortest_path(graph, source=y.item(), target=y_goal.item())
            path = torch.tensor(path, device=y.device)
        except networkx.NetworkXNoPath:
            path = None
        return path
