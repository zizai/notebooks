import matplotlib.pyplot as plt
import scipy.sparse as sps
import torch

from neuroblast.models import SparseLinear


def main():
    x_len = 300
    x_dim = 512

    x = torch.randn(x_len, x_dim)

    y_hat = []
    z_hat = []
    num_samples = 1000
    num_layers = 3
    for _ in range(num_samples):
        w = torch.rand(x_dim, x_dim)
        layers = [torch.nn.Linear(x_dim, x_dim, bias=False)] * num_layers
        layers = torch.nn.Sequential(*layers)
        y = layers(x).sum(0).mean(0).tolist()
        y_hat.append(y)

        layers = [SparseLinear(init_dim=8, kron_k=3, bias=False)] * num_layers
        z = sps.coo_matrix(x.numpy())
        for layer in layers:
            z = layer(z)
        z = z.sum(axis=0).mean(axis=-1)[0, 0]
        z_hat.append(z)

        print(y, z)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[10, 30])
    ax1.hist(y_hat)
    ax2.hist(z_hat)
    ax3.hist2d(y_hat, z_hat)
    plt.show()


if __name__ == '__main__':
    main()
