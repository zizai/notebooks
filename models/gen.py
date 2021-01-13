import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
https://github.com/Diego999/pyGAT

LanczosNet: Multi-Scale Deep Graph Convolutional Networks
https://arxiv.org/abs/1901.01484
https://github.com/lrjconan/LanczosNetwork

Hierarchical Graph Representation Learning with Differentiable Pooling
https://github.com/RexYing/diffpool

Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
https://arxiv.org/abs/1905.07953
https://github.com/benedekrozemberczki/ClusterGCN

FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling
https://arxiv.org/abs/1801.10247
https://github.com/matenure/FastGCN

https://github.com/huangwb/AS-GCN
"""


class NodeModule(nn.Module):
    def __init__(self):
        super(NodeModule, self).__init__()
        self.fc0 = nn.Linear(in_features=512, out_features=512)
        self.fc0bn = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(in_features=512, out_features=254)
        self.fc1bn = nn.BatchNorm1d(254)

    def forward(self, x):
        x = F.relu(self.fc0bn(self.fc0(x)))
        return self.fc1bn(self.fc1(x))


class EdgeModule(nn.Module):
    def __init__(self):
        super(EdgeModule, self).__init__()
        self.fc0 = nn.Linear(in_features=512, out_features=512)
        self.fc0bn = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc1bn = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.fc0bn(self.fc0(x)))
        return self.fc1bn(self.fc1(x))


# Pool is re-used in Baseline and GENs
class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv1bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv2bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256 + 7, 256, kernel_size=3, stride=1, padding=1)
        self.conv5bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256 + 7, 128, kernel_size=3, stride=1, padding=1)
        self.conv6bn = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7bn = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 254, kernel_size=1, stride=1)
        self.conv8bn = nn.BatchNorm2d(254)
        self.pool = nn.AvgPool2d(16)

    def forward(self, x, v):
        # Residual connection
        skip_in = F.relu(self.conv1bn(self.conv1(x)))
        skip_out = F.relu(self.conv2bn(self.conv2(skip_in)))

        r = F.relu(self.conv3bn(self.conv3(skip_in)))
        r = F.relu(self.conv4bn(self.conv4(r))) + skip_out

        # Broadcast
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)

        # Residual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out = F.relu(self.conv5bn(self.conv5(skip_in)))

        r = F.relu(self.conv6bn(self.conv6(skip_in)))
        r = F.relu(self.conv7bn(self.conv7(r))) + skip_out
        r = F.relu(self.conv8bn(self.conv8(r)))

        # Pool
        r = self.pool(r)

        return r


class GraphStructure():

    def __init__(self, num_scenes_per_dim=1, shift=(0, 0)):
        # Initialize nodes and edges, grid structure and interpolator placeholders
        self.EPS = np.exp(-10)
        self.width_one_scene = 2.0
        self.shift = shift
        self.msg_steps = 9
        self.node_dim = 256
        self.update_sz = 254
        self.msg_sz = 256
        self.node_interpolator = {}
        self.initialize_nodes(num_scenes_per_dim)
        self.initialize_edges()
        return

    # Build the list of all node positions within this grid Structure
    def initialize_nodes(self, num_scenes_per_dim=1):
        self.grid = {}
        self.grid['X'] = self.grid['Y'] = num_scenes_per_dim * self.width_one_scene
        self.grid['min_X'] = -1. + self.shift[0]
        self.grid['min_Y'] = -1. + self.shift[1]
        self.grid['n_x'] = self.grid['n_y'] = 4 * (num_scenes_per_dim - 1) + 5
        self.grid['dx'] = self.grid['dy'] = self.grid['X'] / (self.grid['n_x'] - 1)

        # Populate node positions
        node_positions = []
        for x in range(self.grid['n_x']):
            for y in range(self.grid['n_y']):
                node_positions.append(
                    np.array([self.grid['min_X'] + x * self.grid['dx'], self.grid['min_Y'] + y * self.grid['dy']]))

        self.node_positions = torch.FloatTensor(np.vstack(node_positions))
        self.num_nodes = self.node_positions.shape[0]
        return

    # Build the list of all directional edges for this grid Structure
    def initialize_edges(self, eps=0.01):
        smallest_dist = 1e9
        for i in range(self.node_positions.shape[0]):
            for j in range(i + 1, self.node_positions.shape[0]):
                smallest_dist = min(smallest_dist, np.linalg.norm(
                    self.node_positions[i] - self.node_positions[j]))

        edges = []
        for i in range(self.node_positions.shape[0]):
            for j in range(i + 1, self.node_positions.shape[0]):
                if (np.linalg.norm(self.node_positions[i] -
                                   self.node_positions[j]) < (1 + eps) * smallest_dist):
                    edges.append([i, j])
                    edges.append([j, i])

        self.edge_sources = [_[0] for _ in edges]
        self.edge_sinks = [_[1] for _ in edges]
        return

    # Given poses, outputs a score tensor indicating how much weight is assigned to each node, per pose
    def get_interpolation_coordinates(self, poses):
        shape = poses.shape

        # Need nx and ny (num units along each dim of grid) to grab the node_positions for each point from input
        # - self.grid['min_X'] or - self.grid['min_Y'] allows yielding positive indices
        nx = ((poses[:, :, 0] - self.grid['min_X']) / self.grid['dx']).floor_().long()
        ny = ((poses[:, :, 1] - self.grid['min_Y']) / self.grid['dy']).floor_().long()
        node_pos = self.node_positions.to(poses.device)

        # Flatten indices tensors to get 1D tensor for use in index_select
        bottom_left_idx = (nx * self.grid['n_y'] + ny).reshape(shape[0] * shape[1])
        bottom_right_idx = (nx * self.grid['n_y'] + ny + 1).reshape(shape[0] * shape[1])
        top_left_idx = ((nx + 1) * self.grid['n_y'] + ny).reshape(shape[0] * shape[1])
        top_right_idx = ((nx + 1) * self.grid['n_y'] + ny + 1).reshape(shape[0] * shape[1])

        # Grab the meaningful nodes
        bottom_left = torch.index_select(node_pos, dim=0, index=bottom_left_idx).reshape(shape[0], shape[1], 2)
        bottom_right = torch.index_select(node_pos, dim=0, index=bottom_right_idx).reshape(shape[0], shape[1], 2)
        top_left = torch.index_select(node_pos, dim=0, index=top_left_idx).reshape(shape[0], shape[1], 2)
        top_right = torch.index_select(node_pos, dim=0, index=top_right_idx).reshape(shape[0], shape[1], 2)

        # Grab original coordinates of input points
        original_xy = poses[:, :, :2]

        # Each point is in a square, which we normalize to width,height (1,1)
        # The weighting of each point is equal to the area of the rectangle
        # between x and the opposite corner.
        dd = torch.FloatTensor([self.grid['dx'], self.grid['dy']]).to(poses.device)
        bottom_left_score = torch.prod(torch.abs(top_right - original_xy) / dd, dim=2)
        bottom_right_score = torch.prod(torch.abs(top_left - original_xy) / dd, dim=2)
        top_left_score = torch.prod(torch.abs(bottom_right - original_xy) / dd, dim=2)
        top_right_score = torch.prod(torch.abs(bottom_left - original_xy) / dd, dim=2)

        # Initialize a matrix of scores for every node_position in grid (for every frame of every batch)
        scores = torch.zeros(shape[0], shape[1], self.node_positions.shape[0]).to(poses.device)

        # Scatter scores along scores tensor for every node_position in grid (for every frame of every batch),
        # based on interpolated node coordinates
        scores.scatter_(dim=2, index=bottom_left_idx.reshape(shape[0], shape[1], 1),
                        src=torch.unsqueeze(bottom_left_score, dim=2))
        scores.scatter_(dim=2, index=bottom_right_idx.reshape(shape[0], shape[1], 1),
                        src=torch.unsqueeze(bottom_right_score, dim=2))
        scores.scatter_(dim=2, index=top_left_idx.reshape(shape[0], shape[1], 1),
                        src=torch.unsqueeze(top_left_score, dim=2))
        scores.scatter_(dim=2, index=top_right_idx.reshape(shape[0], shape[1], 1),
                        src=torch.unsqueeze(top_right_score, dim=2))

        return scores


class GEN(nn.Module):
    def __init__(self, num_copies=0):
        super(GEN, self).__init__()

        self.structure = GraphStructure()
        self.node_module = NodeModule()
        self.edge_module = EdgeModule()
        self.embedder = Pool()

        self.num_copies = num_copies

    # Refreshes self.structure when graph shape changes
    def refresh_structure(self, num_scenes_per_dim, shift):
        self.structure = GraphStructure(num_scenes_per_dim=num_scenes_per_dim, shift=shift)
        return

    # Preprocesses raw frames and poses into embeddings which then fit into the node hidden states
    # using an attention score interpolated from the structure of the graph and position of the inputs
    def inp_to_graph_inp(self, view_frames, view_poses):
        batch_size, num_views_per_batch = view_frames.shape[:2]

        frames_reshaped = view_frames.reshape(batch_size * num_views_per_batch, 3, 64, 64)
        poses_reshaped = view_poses.reshape(batch_size * num_views_per_batch, 7)
        embeddings = self.embedder(frames_reshaped, poses_reshaped)
        embeddings = embeddings.reshape(batch_size, num_views_per_batch, 254)

        # Populate state of each node (for each batch)
        # 1) Organize node_positions
        node_positions = self.structure.node_positions.unsqueeze(0).repeat((batch_size, 1, 1)).to(view_frames.device)
        # 2) Get weighted embeddings per node
        scores = self.structure.get_interpolation_coordinates(view_poses)
        # Reshape for bmm
        scores = scores.permute(0, 2, 1)
        # Get weighted embeddings
        weighted = torch.bmm(scores, embeddings)
        # 3) Concat node positions to the weighted embeddings to get tail shape 2+254
        inp = torch.cat([node_positions, weighted], dim=2)
        return inp

    # Given node hidden states, extrapolates query embeddings from the graph
    # based on the positions of the queries
    def graph_out_to_out(self, nodes_states, query_poses):
        batch_size, poses_per_scene, pose_dimension = query_poses.shape

        # F is the nodes hidden states per scene (per room or per maze configuration)
        assert (nodes_states.shape == (batch_size, self.structure.num_nodes, self.structure.node_dim))
        # Interpolation tensor
        attn = self.structure.get_interpolation_coordinates(query_poses)
        # Extract the weighted sum of node hidden states to decode, for each pose
        extraction = torch.bmm(attn, nodes_states)
        # First two coordinates: query_poses added even when num_copies=0
        extraction = torch.cat([query_poses] * self.num_copies + [extraction], dim=2)
        return extraction

    # Given raw frames and poses, embeds their information into node hidden states,
    # performs message passing along the graph using node and edge modules,
    # and outputs an extracted embedding based on the position of queries
    def forward(self, view_frames, view_poses, query_poses):
        bs = view_frames.shape[0]

        nodes_states = self.inp_to_graph_inp(view_frames, view_poses)

        edge_sources = torch.tensor(self.structure.edge_sources, dtype=torch.int).to(view_frames.device)
        edge_sinks = torch.tensor(self.structure.edge_sinks, dtype=torch.int).to(view_frames.device)

        # Perform message passing along edges
        for step in range(self.structure.msg_steps):
            sources = nodes_states[:, edge_sources, :].clone()
            sinks = nodes_states[:, edge_sinks, :].clone()

            inp = torch.cat([sources, sinks], dim=2)
            out = self.edge_module(inp.view(-1, inp.shape[2])).view(bs, -1, self.structure.msg_sz)

            incoming_msgs = torch.zeros(bs, self.structure.num_nodes, self.structure.msg_sz).to(view_frames.device)
            incoming_msgs = incoming_msgs.index_add(1, edge_sinks, out)

            # Update node hidden states based on messages received
            msgs_and_nodes_states = torch.cat([incoming_msgs, nodes_states], dim=2).view(
                -1, self.structure.node_dim + self.structure.msg_sz)
            update = self.node_module(msgs_and_nodes_states).view(bs, -1, self.structure.update_sz)
            nodes_states[:, :, -self.structure.update_sz:] += update

        # Return embeddings extracted from final node hidden states based on query positions
        return self.graph_out_to_out(nodes_states, query_poses)
