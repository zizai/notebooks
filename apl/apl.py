import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


class Prior(nn.Module):
    """
    Maps a state s to mu and sigma which will define the generative model
    from which we sample new states.

    Parameters
    ----------
    s_dim : int
        Dimension of state representation s.
    """
    def __init__(self, s_dim):
        super(Prior, self).__init__()
        layers = [
            nn.Linear(s_dim, s_dim),
            nn.ReLU(inplace=True),
            nn.Linear(s_dim, s_dim),
            nn.ReLU(inplace=True),
            nn.Linear(s_dim, s_dim),
            nn.ReLU(inplace=True),
        ]
        self.s_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu_s = nn.Linear(s_dim, s_dim)
        self.hidden_to_sigma_s = nn.Linear(s_dim, s_dim)

    def forward(self, s):
        hidden = self.s_to_hidden(s)

        # p(s_t+1|s_t)
        mu_s = self.hidden_to_mu_s(hidden)
        sigma_s = 1e-6 + F.softplus(self.hidden_to_sigma_s(hidden))
        p_s = Normal(mu_s, sigma_s)

        return p_s


class Posterior(nn.Module):
    """
    Maps past state s_tm1 and current observation o_t to the current state s_t.

    Parameters
    ----------
    s_dim : int
        Dimension of state representation s.

    o_dim : int
        Dimension of observation o.
    """
    def __init__(self, s_dim, o_dim, a_dim):
        super(Posterior, self).__init__()
        self.o_to_s = nn.GRUCell(o_dim + a_dim, s_dim)
        self.s_to_mu = nn.Linear(s_dim, s_dim)
        self.s_to_sigma = nn.Linear(s_dim, s_dim)

    def forward(self, s_tm1, o_t, a_t):
        """
        Parameters
        ----------
        s_tm1 : torch.Tensor
            Shape (batch_size, s_dim)

        o_t : torch.Tensor
            Shape (batch_size, o_dim)
        """
        input = torch.cat((o_t, a_t), dim=1)
        s_t = self.o_to_s(input, s_tm1)
        mu = self.s_to_mu(s_t)
        sigma = 1e-6 + F.softplus(self.s_to_sigma(s_t))
        q_s = Normal(mu, sigma)
        return q_s


class Likelihood(nn.Module):
    """
    Maps a state to an observation
    """
    def __init__(self, s_dim, o_dim, a_dim):
        super(Likelihood, self).__init__()
        layers = [
            nn.Linear(s_dim, s_dim),
            nn.ReLU(inplace=True),
            nn.Linear(s_dim, s_dim),
            nn.ReLU(inplace=True),
        ]
        self.s_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu_o = nn.Linear(s_dim, o_dim)
        self.hidden_to_sigma_o = nn.Linear(s_dim, o_dim)
        self.hidden_to_mu_a = nn.Linear(s_dim, a_dim)
        self.hidden_to_sigma_a = nn.Linear(s_dim, a_dim)

    def forward(self, s_t):
        hidden = self.s_to_hidden(s_t)

        # p(o_t|s_t)
        mu_o = self.hidden_to_mu_o(hidden)
        sigma_o = 1e-6 + F.softplus(self.hidden_to_sigma_o(hidden))
        p_o = Normal(mu_o, sigma_o)

        # p(a_t|s_t)
        mu_a = self.hidden_to_mu_a(hidden)
        sigma_a = 1e-6 + F.softplus(self.hidden_to_mu_a(hidden))
        p_a = Normal(mu_a, sigma_a)

        return p_o, p_a


class Control(nn.Module):
    """
    Maps current state to action.
    Similar to policy function in RL.

    Parameters
    ----------
    s_dim : int
        Dimension of state representation s.

    a_dim : int
        Dimension of action a.
    """
    def __init__(self, s_dim, a_dim):
        super(Control, self).__init__()
        layers = [
            nn.Linear(s_dim, s_dim),
            nn.Tanh(),
        ]
        self.s_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu_a = nn.Linear(s_dim, a_dim)
        self.hidden_to_sigma_a = nn.Linear(s_dim, a_dim)

    def forward(self, s_t):
        hidden = self.s_to_hidden(s_t)
        mu_a = self.hidden_to_mu_a(hidden)
        sigma_a = 1e-6 + F.softplus(self.hidden_to_sigma_a(hidden))
        a_tp1 = Normal(mu_a, sigma_a).rsample()

        return a_tp1


class APL(nn.Module):
    """
    Action Perception Loop

    Parameters
    ----------
    s_dim : int
        Dimension of state representation s.

    o_dim : int
        Dimension of observation o.

    a_dim : int
        Dimension of action a.
    """
    def __init__(self, s_dim, o_dim, a_dim):
        super(APL, self).__init__()

        self.prior = Prior(s_dim)
        self.posterior = Posterior(s_dim, o_dim, a_dim)
        self.likelihood = Likelihood(s_dim, o_dim, a_dim)
        self.control = Control(s_dim, a_dim)

    def forward(self, s_tm1, o_t, a_t):
        p_s = self.prior(s_tm1)
        q_s = self.posterior(s_tm1, o_t, a_t)

        s_t = q_s.rsample([20])

        p_o, p_a = self.likelihood(s_t)
        a_tp1 = self.control(s_t)

        return s_t, a_tp1, q_s, p_s, p_o, p_a
