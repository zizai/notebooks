import torch

from neuroblast.models.vae import VGAE


x_dim = 512
num_nodes = 300
x = torch.randn((num_nodes, x_dim))


def test_vgae():
    vae = VGAE(x_dim).requires_grad_(False)

    r, edge_index = vae.encode(x)
    print(r.shape, edge_index.shape)

    z_hat = vae.reparameterize(r.sum(0)).rsample()
    print(torch.norm(r), torch.norm(r.sum(0)), torch.norm(z_hat))

    z = vae.sample_prior(r.size(0))
    print(z.shape)

    x_hat, _ = vae.decode(z)
    print(x_hat.shape)

    x_hat, edge_index = vae.decode(r, edge_index)
    print(x_hat.shape, edge_index.shape)
