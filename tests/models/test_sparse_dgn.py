import numpy as np
import scipy.sparse as sps

from neuroblast.models.sparse_dgn import NodeReplication, NodeAssembly, GraphEncoder, GraphDecoder

x_dim = 32
s_num_nodes = 3
l_num_nodes = 100
k = 3


def test_node_assembly():
    pool = NodeAssembly(x_dim)

    x = np.random.randn(l_num_nodes, x_dim)
    edge_index = sps.rand(x.shape[0], x.shape[0], density=0.1).nonzero()
    edge_index = np.array(edge_index)

    print(x.shape, edge_index.shape)
    print(np.linalg.norm(x))
    x, edge_index = pool.forward(x, edge_index)
    x, edge_index = pool.forward(x, edge_index)
    print(x.shape, edge_index.shape)
    print(np.linalg.norm(x))

    x = np.random.randn(s_num_nodes, x_dim)
    edge_index = sps.rand(x.shape[0], x.shape[0], density=0.7).nonzero()
    edge_index = np.array(edge_index)

    print(x.shape, edge_index.shape)
    x, edge_index = pool.forward(x, edge_index)
    print(x.shape, edge_index.shape)


def test_node_replication():
    unpool = NodeReplication(x_dim)

    x = np.random.randn(l_num_nodes, x_dim)
    edge_index = sps.rand(x.shape[0], x.shape[0], density=0.01).nonzero()
    edge_index = np.array(edge_index)

    print(x.shape, edge_index.shape)
    x, edge_index = unpool.forward(x, edge_index)
    print(x.shape, edge_index.shape)

    x = np.random.randn(s_num_nodes, x_dim)
    edge_index = sps.rand(x.shape[0], x.shape[0], density=0.7).nonzero()
    edge_index = np.array(edge_index)

    print(x.shape, edge_index.shape)
    x, edge_index = unpool.forward(x, edge_index)
    print(x.shape, edge_index.shape)


def test_graph_encoder():
    x = np.random.randn(l_num_nodes, x_dim)
    adj = sps.rand(x.shape[0], x.shape[0], density=0.1)
    edge_index = np.array(adj.nonzero())

    enc = GraphEncoder(x_dim, density=10/x_dim)

    print(x.shape, edge_index.shape)
    x, edge_index = enc.forward(x, edge_index)
    print(x.shape, edge_index.shape)


def test_graph_decoder():
    x = np.random.randn(s_num_nodes, x_dim)
    adj = sps.rand(x.shape[0], x.shape[0], density=1)
    edge_index = np.array(adj.nonzero())

    dec = GraphDecoder(x_dim, density=10/x_dim)

    print(x.shape, edge_index.shape)
    x, edge_index = dec.forward(x, edge_index)
    print(x.shape, edge_index.shape)
