import torch
from neuroblast.agents.utils import (convert_seq_to_graph)


def test_convert_seq_to_graph():
    v = [torch.tensor([i for i in range(3)]), torch.tensor([i for i in range(5)])]
    graphs = convert_seq_to_graph(v, max_node_degree=3)
    assert graphs[0].x.shape[0] == 8
    assert graphs[0].edge_index.shape[1] == 21
