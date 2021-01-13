import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GraphDiscriminator(nn.Module):
    def __init__(self, x_dim, r_dim, n_layers=3, n_heads=8):
        super(GraphDiscriminator, self).__init__()
        self.embed = nn.Linear(x_dim, r_dim, bias=False)

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            assert r_dim % n_heads == 0
            self.convs.append(gnn.GATConv(r_dim, r_dim, n_heads, concat=False))

        self.fc = nn.Linear(2 * r_dim, 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor, c: torch.Tensor):
        x = self.embed(x)
        for i in range(self.n_layers):
            x = self.convs[i](x, e)
            x = F.relu(x)
        xc = torch.cat((x.sum(dim=0, keepdim=True), c.sum(dim=0, keepdim=True)), dim=-1)
        logits = torch.sigmoid(self.fc(xc))
        return logits


class TransformerDiscriminator(nn.Module):
    def __init__(self, r_dim, n_layers=3, n_heads=8):
        super(TransformerDiscriminator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(r_dim, n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(2 * r_dim, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        r = self.encoder(x).sum(dim=0)
        rc = torch.cat((r, c), dim=-1)
        logits = torch.sigmoid(self.fc(rc))
        return logits


class InfomaxDiscriminator(nn.Module):
    def __init__(self, n_h):
        super(InfomaxDiscriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
