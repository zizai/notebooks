import numpy as np
import time
import torch
from torch_geometric.nn import GCNConv

from scipy import sparse

from torch_geometric.utils import dense_to_sparse


def main():
    x_dim = 512
    x_len = 10000

    x = sparse.rand(x_len, x_dim, density=10/x_dim, format='csr', dtype=np.float)
    adj = sparse.rand(x_len, x_len, density=10/x_len, format='csr', dtype=np.float)
    w = sparse.rand(x_dim, x_dim, density=10/x_dim, format='csr', dtype=np.float)

    start = time.time()
    adj.dot(x.dot(w))
    print(time.time() - start)

    x1 = x.todense().astype(np.float)
    adj1 = adj.todense().astype(np.float)
    w1 = w.todense().astype(np.float)

    start = time.time()
    adj1.dot(x1.dot(w1))
    print(time.time() - start)

    x2 = torch.tensor(x1, dtype=torch.float)
    adj2 = torch.tensor(adj1, dtype=torch.float)
    w2 = torch.tensor(w1, dtype=torch.float)

    start = time.time()
    adj2.matmul(x2.matmul(w2))
    print(time.time() - start)

    adj2alt = torch.rand((x_len, x_len), dtype=torch.float)
    start = time.time()
    adj2alt.matmul(x2.matmul(w2))
    print(time.time() - start)

    conv = GCNConv(x_dim, x_dim)
    edge_index, _ = dense_to_sparse(adj2)

    start = time.time()
    x3 = conv(x2, edge_index)
    print(time.time() - start)


if __name__ == '__main__':
    # main()
    start = time.time()
    x = sps.randn(1000, 1000, density=0.05)
    for i in range(1000):
        x += x
    print(time.time() - start)

    start = time.time()
    x = torch.randn((1000, 1000))
    for i in range(1000):
        x += x
    print(time.time() - start)
