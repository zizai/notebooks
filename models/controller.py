import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import RelaxedOneHotCategoricalStraightThrough


class Controller(nn.Module):
    def __init__(self, device, s_dim, a_dim):
        super(Controller, self).__init__()

        self.device = device

        layers = [
            nn.Linear(s_dim, s_dim),
            nn.ReLU(),
        ]
        self.s_to_hidden = nn.Sequential(*layers)
        self.hidden_to_scores = nn.Linear(s_dim, a_dim)

    def forward(self, s_t):
        hidden = self.s_to_hidden(s_t)
        probs = F.softmax(self.hidden_to_scores(hidden), dim=-1)
        a_t = RelaxedOneHotCategoricalStraightThrough(0.5, probs=probs).rsample()

        return a_t
