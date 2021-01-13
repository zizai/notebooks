import torch
from torch.distributions.kl import kl_divergence


class Agent(object):
    def _kl_divergence(self, q_s, p_s):
        if p_s is None:
            # assume p_s is standard normal
            mu_1 = q_s.loc
            var_1 = q_s.scale
            return 0.5 * (-torch.log(var_1) + var_1 + mu_1 ** 2 - 1)
        else:
            return kl_divergence(q_s, p_s)

