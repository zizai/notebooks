import math

import torch


def seq_to_graph(x, stride=None):
    seq_len = x.size(0)
    if stride is None:
        stride = max(math.ceil(math.log(seq_len)), 1)

    src = []
    tgt = []
    for i in range(seq_len):
        num_neighbors = min(stride, seq_len - i)
        src += [i] * num_neighbors
        tgt += [i + k for k in range(num_neighbors)]

    edge_index = torch.tensor([src, tgt], device=x.device, dtype=torch.long)
    return edge_index
