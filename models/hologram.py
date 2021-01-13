import math
import time

import networkx
import torch
from models.dgn import GraphDecoderLayer
from torch import nn
from torch.nn import Linear


class Genotype:
    def __init__(self, z_dim=1024):
        self.z_dim = z_dim
        self.p_z = torch.rand(self.z_dim)
        self.z = self.sample()

    def sample(self):
        return torch.bernoulli(self.p_z)


class Decoder(nn.Module):
    def __init__(self, r_dim, num_layers=3, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        layer = GraphDecoderLayer(r_dim, dropout=dropout)
        self.layers = nn.ModuleList([layer] * self.num_layers)
        self.input_map = nn.Linear(r_dim, 1)

    def forward(self, x, edge_index):
        for _, layer in enumerate(self.layers):
            x, edge_index = layer(x, edge_index)
        source_prob = torch.sigmoid(self.input_map(x)).reshape(-1)
        source_index = torch.nonzero(torch.bernoulli(source_prob)).reshape(-1).tolist()
        if not source_index:
            source_index = [0]
        return x, edge_index, source_index


class Phenotype:
    def __init__(self, cells: list, graph: networkx.DiGraph, source_index: list):
        assert len(cells) > 0
        assert graph.number_of_nodes() > 0
        assert len(source_index) > 0
        self.cells = cells
        self.graph = graph
        self.source_index = source_index
        self.traversal_depth = 3
        self.num_of_nodes = len(cells)
        self.num_of_edges = graph.number_of_edges()

    def propagate(self, messages: torch.Tensor, source_index: list, activation=torch.relu):
        out = []
        all_nbr = []
        for i in range(len(source_index)):
            node_id = source_index[i]
            r = self.cells[node_id].forward(messages[i]).unsqueeze(0)
            if activation:
                r = activation(r)
            if self.graph.has_node(node_id):
                nbr = [k for k in self.graph[node_id].keys()]
                if nbr:
                    # send message to neighbors
                    out.append(torch.cat([r] * len(nbr)))
                    all_nbr.append(torch.as_tensor(nbr))
                else:
                    # return message to itself
                    out.append(r)
                    all_nbr.append(torch.as_tensor([node_id]))
            else:
                # return message to itself
                out.append(r)
                all_nbr.append(torch.as_tensor([node_id]))

        out = torch.cat(out)
        all_nbr = torch.cat(all_nbr)

        all_nbr, sorted_index = torch.sort(all_nbr)
        out = out[sorted_index]

        pointer = all_nbr[0]
        group = []
        reduced_out = []
        next_index = []
        for i, next_nbr in enumerate(all_nbr):
            if pointer == next_nbr and next_nbr != all_nbr[-1]:
                group.append(i)
            elif pointer == next_nbr and next_nbr == all_nbr[-1]:
                group.append(i)
                group = torch.as_tensor(group)
                reduced_out.append(out[group].mean(0, True))
                next_index.append(pointer)
            elif pointer != next_nbr and next_nbr == all_nbr[-1]:
                group = torch.as_tensor(group)
                reduced_out.append(out[group].mean(0, True))
                next_index.append(pointer)

                group = torch.as_tensor([i])
                reduced_out.append(out[group])
                next_index.append(next_nbr)
            else:
                group = torch.as_tensor(group)
                reduced_out.append(out[group].mean(0, True))
                next_index.append(pointer)
                pointer = next_nbr
                group = [i]

        reduced_out = torch.cat(reduced_out)

        # out = scatter_('add', out, all_nbr)
        # print(out.shape, self.num_of_nodes)
        # reduced_out = []
        # next_index = []
        # for i, batch in enumerate(out):
        #     if batch.sum() != 0:
        #         reduced_out.append(batch.unsqueeze(0))
        #         next_index.append(i)
        #
        # reduced_out = torch.cat(reduced_out)
        # print(reduced_out.shape, len(next_index))

        return reduced_out, next_index

    def act(self, x: torch.Tensor):
        next_index = self.source_index
        messages = torch.cat([x.unsqueeze(0)] * len(next_index))
        activations = [torch.relu] * (self.traversal_depth - 1) + [None]
        for i in range(self.traversal_depth):
            messages, next_index = self.propagate(messages, next_index, activations[i])

        return messages


class Hologram(nn.Module):
    def __init__(self, in_dim, out_dim, activation=torch.tanh, genome=None):
        super(Hologram, self).__init__()
        self.activation = activation
        self.z_dim = in_dim ** 2
        self.w_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # initialize genotype
        if genome is not None:
            assert genome.z_dim == self.z_dim
            self.genome = genome
        else:
            self.genome = Genotype(self.z_dim)

        # set up decoder
        self.decoder = Decoder(self.z_dim, num_layers=1)
        self.fc = nn.Linear(in_dim, out_dim)

        # initialize phenotype
        self.phenom = self.gen()

    def act(self, obs):
        out = self.phenom.act(obs).mean(0)
        return self.activation(self.fc(out))

    def gen(self, genome: Genotype = None) -> Phenotype:
        """
            generate neural phenotypes
        Args:
            genome:

        Returns:
            phenom: computation graph and parameters
        """

        # epi-genetics
        if genome:
            self.genome = genome
        x = self.genome.z.reshape(1, -1)
        edge_index = torch.as_tensor([[0], [0]])
        for i in range(10):
            x, edge_index, source_index = self.decoder.forward(x, edge_index)

        # network morphism
        cells = []
        for row in x:
            w = row.reshape(self.w_dim, self.w_dim)
            linear = nn.Linear(self.w_dim, self.w_dim, bias=False).requires_grad_(False)
            linear.weight.data = w
            cells.append(linear)
        edge_list = edge_index.reshape(-1, 2).tolist()
        graph = networkx.DiGraph(edge_list)
        phenom = Phenotype(cells, graph, source_index)

        return phenom

    def update(self):
        self.phenom = self.gen()

    def learn(self, data):
        """
            learn new phenotypes from data
        Args:
            data:

        Returns:

        """


def main():
    input_dim = 32
    output_dim = 32
    hg = Hologram(input_dim, output_dim)

    genome = Genotype()
    with torch.no_grad():
        phenom = hg.gen(genome)
    print("number of cells: {}".format(phenom.num_of_nodes))
    print("number of edges: {}".format(phenom.num_of_edges))
    print(genome.z.norm(), phenom.cells[0].weight.norm())

    x = torch.randn((10, input_dim))

    start = time.time()
    out = phenom.act(x)
    out = hg.out(out)
    print(time.time() - start)
    print(x.mean(0))
    print(out.mean(0).mean(0))

    layers = [nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim), nn.ReLU()]
    fc = nn.Sequential(*layers)
    print([p.data.norm() for p in fc.parameters()])
    start = time.time()

    out = fc.forward(x)
    print(time.time() - start)

    print(out.norm())


if __name__ == '__main__':
    main()
