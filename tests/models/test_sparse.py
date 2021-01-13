import time

import networkx as nx
import numpy as np
import scipy.sparse as sps
import torch
from torch.nn import Linear
from torch.nn.init import xavier_uniform_

from neuroblast.models.sparse import SparseEmbedding, SparseGCN, SparseLinear
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.utils import dense_to_sparse


def test_embedding():
    emb_dim = 50000
    emb = SparseEmbedding(emb_dim)

    tokens = np.random.randint(0, emb_dim, 10)
    output = emb.forward(tokens)
    assert output.tocsr().sum() == len(tokens)
    print(output.shape)


def test_gcn():
    x_len, x_dim = 100, 1000
    x = np.random.randn(x_len, x_dim)
    adj = sps.rand(x_len, x_len, density=0.1)
    edge_index = np.array(adj.nonzero())

    gcn = SparseGCN(x_dim, x_dim)
    print(x[7].mean(), np.linalg.norm(x))

    start = time.time()
    out = gcn.forward(x, edge_index)
    print(time.time() - start)
    print(out[7].mean(), np.linalg.norm(out), sps.linalg.norm(gcn.w))

    gcn1 = DenseGCNConv(x_dim, x_dim, improved=True, bias=False)
    adj = adj > 0
    out = gcn1(torch.tensor(x, dtype=torch.float), torch.tensor(adj.toarray(), dtype=torch.float))
    print(out[0, 7].mean(), out.norm(), gcn1.weight.norm())


def test_linear():
    x_len, x_dim = 100, 1000
    x = np.random.randn(x_len, x_dim)

    linear = SparseLinear(x_dim, x_dim, bias=True)

    print(np.linalg.norm(x))
    out = linear.forward(x)
    print(np.linalg.norm(out), sps.linalg.norm(linear.weight))

    linear1 = Linear(x_dim, x_dim)
    xavier_uniform_(linear1.weight)
    out1 = linear1(torch.tensor(x, dtype=torch.float))
    print(out1.norm(), linear1.weight.norm())
