import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveNodeFeatureCoupling(nn.Module):
    def __init__(self):
        super(AdditiveNodeFeatureCoupling, self).__init__()


class AdditiveAdjCoupling(nn.Module):
    def __init__(self):
        super(AdditiveAdjCoupling, self).__init__()


class GraphNVP(nn.Module):
    def __init__(self, flow_depth):
        super(GraphNVP, self).__init__()
        self.flow_depth = flow_depth
        channel_coupling = AdditiveNodeFeatureCoupling
        node_coupling = AdditiveAdjCoupling

        self.channel_layers = nn.ModuleList([
            channel_coupling()
            for i in range(flow_depth)
        ])

        self.node_layers = nn.ModuleList([
            node_coupling()
            for i in range(flow_depth)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        h = x + torch.empty_like(x).uniform_(0., 0.9)
        adj = adj + torch.empty_like(adj).uniform_(0., 0.9)
        sum_log_det_x = 0
        sum_log_det_adj = 0

        for i in range(self.flow_depth):
            h, log_det_x = self.channel_layers[i](h, adj)
            adj, log_det_adj = self.node_layers[i](adj)
            sum_log_det_x += log_det_x
            sum_log_det_adj += log_det_adj

        return h, adj, sum_log_det_x, sum_log_det_adj

    def reverse(self, z: torch.Tensor, adj: torch.Tensor) -> tuple:
        for i in reversed(range(self.flow_depth)):
            adj, log_det_adj = self.node_layers[i](adj)

