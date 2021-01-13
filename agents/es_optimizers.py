# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from agents.utils import SharedNoiseTable


def compute_ranks(x):
    """Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in
    [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(
            itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(
            np.asarray(batch_weights, dtype=np.float32),
            np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


class Optimizer(object):
    def __init__(self, model, noise, l2_coeff):
        self.model = model
        self.noise = SharedNoiseTable(noise)
        self.l2_coeff = l2_coeff
        self.num_params = model.get_weights().size
        self.t = 0

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, theta):
        self.model.set_weights(theta)

    def update(self, noisy_returns, noise_indices):
        self.t += 1

        # Process the returns
        proc_noisy_returns = compute_centered_ranks(noisy_returns)

        # Compute and take a step.
        g, count = batched_weighted_sum(
            proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
            (self.noise.get(index, self.num_params) for index in noise_indices),
            batch_size=500)
        g /= noisy_returns.size
        assert (g.shape == (self.num_params, ) and g.dtype == np.float32 and count == len(noise_indices))

        theta = self.get_weights()
        globalg = - g + self.l2_coeff * theta
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        theta = theta + step
        self.set_weights(theta)
        return theta, ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, model, noise, l2_coeff, lr, momentum=0.9):
        Optimizer.__init__(self, model, noise, l2_coeff)
        self.v = np.zeros(self.num_params, dtype=np.float32)
        self.lr, self.momentum = lr, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.lr * self.v
        return step


class Adam(Optimizer):
    def __init__(self, model, noise, l2_coeff, lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, model, noise, l2_coeff)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.num_params, dtype=np.float32)
        self.v = np.zeros(self.num_params, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.lr * (np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
