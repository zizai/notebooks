import numpy as np
import torch
from neuroblast.agents.molda import MoldaModel
from neuroblast.agents.es_optimizers import SGD, Adam
from torch_geometric.data import Data


x = [[0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0]]
edge_index = [[0, 1, 2, 3, 0, 0, 1, 1, 3, 3],
              [1, 2, 3, 0, 4, 5, 6, 7, 8, 9]]

x = torch.tensor(x, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)
g = Data(x=x, edge_index=edge_index)

model = MoldaModel(x.size(1))
model.requires_grad_(False)

seed = 123
count = 10000000
noise = np.random.RandomState(seed).randn(count).astype(np.float32)

noise_stdev = 0.02
l2_coeff = 0.005
lr = 0.01


def rollout(optim, timesteps):
    noisy_returns = []
    noise_indices = []

    for t in range(timesteps):
        theta = optim.get_weights()

        noise_index = optim.noise.sample_index(optim.num_params)

        perturbation = noise_stdev * optim.noise.get(noise_index, optim.num_params)

        optim.set_weights(theta + perturbation)
        results_pos = model.discriminator_step(g)
        reward_pos = - results_pos[-1].numpy()

        optim.set_weights(theta - perturbation)
        results_neg = model.discriminator_step(g)
        reward_neg = - results_neg[-1].numpy()

        optim.set_weights(theta)

        noise_indices.append(noise_index)
        noisy_returns.append([reward_pos, reward_neg])

    noisy_returns = np.array(noisy_returns)
    noise_indices = np.array(noise_indices)
    return noisy_returns, noise_indices


def test_sgd():
    optim = SGD(model.discriminator, noise, l2_coeff, lr)

    noisy_returns, noise_indices = rollout(optim, 10)
    theta, update_ratio = optim.update(noisy_returns, noise_indices)
    print(update_ratio)


def test_adam():
    optim = Adam(model.discriminator, noise, l2_coeff, lr)
    noisy_returns, noise_indices = rollout(optim, 10)
    theta, update_ratio = optim.update(noisy_returns, noise_indices)
    print(update_ratio)
