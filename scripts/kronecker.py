import math

import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sps


def main():
    init_dim = 8
    kron_k = 4
    g = nx.erdos_renyi_graph(init_dim, 2/init_dim, seed=1234, directed=True)
    adj = nx.adjacency_matrix(g)

    fig, axs = plt.subplots(kron_k-1, 2, constrained_layout=False, figsize=[20, 10*(kron_k-1)])
    for k in range(kron_k-1):
        g = nx.erdos_renyi_graph(init_dim, 2 / init_dim, seed=k*1000, directed=True)
        adj1 = nx.adjacency_matrix(g)
        adj = sps.kron(adj, adj1)
        axs[k, 0].matshow(adj.todense())
        axs[k, 1].matshow(adj1.todense())

    plt.show()


if __name__ == '__main__':
    main()
