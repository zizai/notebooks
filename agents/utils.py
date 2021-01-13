import math
import numpy as np
import torch
from constants import (MAX_NODE_DEGREE, MAX_SEQ_LEN)
from torch_geometric.data import (Batch, Data)


class SharedNoiseTable(object):
    def __init__(self, noise):
        self.noise = noise
        assert self.noise.dtype == np.float32

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)


def tanh_to_action(activation, min_, max_):
    activation = np.clip(activation, -1, 1)
    return (activation + 1) / 2 * (max_ - min_) + min_


def normalize_obs(obs, min_, max_):
    return 2 * (obs - min_) / (max_ - min_) - 1


def convert_seq_to_graph(v_t, max_graph_size=1024, max_node_degree=MAX_NODE_DEGREE):
    vertices = torch.cat(v_t, dim=-1)
    vertex_count = vertices.shape[0]
    num_parts = math.ceil(vertex_count / max_graph_size)
    vertex_parts = vertices.chunk(num_parts, dim=0)
    graphs = []

    for b, vertex_part in enumerate(vertex_parts):
        vertex_part_count = vertex_part.shape[0]
        edges = []

        for i in range(vertex_part_count):
            v_i_degree = min(max_node_degree, vertex_part_count - i)
            edges += [[i, i + j] for j in range(v_i_degree)]

        edges = torch.tensor(edges, dtype=torch.long).transpose(0, 1)
        g = Data(x=vertex_part, edge_index=edges)
        graphs.append(g)

    return graphs


def unpack_kv(obs, k_dim, max_seq_len=MAX_SEQ_LEN):
    k = []
    v = []
    for each_obs in obs:
        try:
            o = torch.tensor(each_obs)
            k_b = o[:max_seq_len, :k_dim]
            v_b = o[:max_seq_len, k_dim:].squeeze(1)
            k.append(k_b)
            v.append(v_b)
        except IndexError:
            print(each_obs)
            raise
    v_len = [line.shape[0] for line in v]
    v_len = torch.tensor(v_len, dtype=torch.long)

    return k, v, v_len


if __name__ == '__main__':
    seq = [torch.tensor([i for i in range(5)]), torch.tensor([i for i in range(11)])]
    graphs = convert_seq_to_graph(seq, 3, 3)
    print(graphs)

