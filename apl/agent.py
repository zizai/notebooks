import gym
import torch
from torch.nn import functional as F
from torch.distributions import (Categorical, Normal)
from torch.distributions.kl import kl_divergence

from apl import APL


class Agent(object):
    def __init__(self, device, env, lr=3e-4, state_dim=64, goal_obs=None):
        self.device = device
        self.env = env
        self.state_dim = state_dim
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] if env.action_space.shape else env.action_space.n
        self.goal_obs = goal_obs
        self.lr = lr

        self.apl = APL(s_dim=self.state_dim, o_dim=self.obs_dim, a_dim=self.action_dim)
        self.optimizer = torch.optim.Adam(self.apl.parameters(), lr=lr)
        self.s_0 = torch.zeros(1, self.state_dim, dtype=torch.float)
        self.a_0 = torch.zeros(1, self.action_dim, dtype=torch.float)
        self.s_t = self.s_0
        self.a_t = self.a_0
        self.q_s = None
        self.p_s = None
        self.p_o = None
        self.p_a = None

    def _calc_FE(self, q_s, p_s, o_t, p_o, a_t, p_a):
        """
        Computes APL Free Energy.
        Parameters
        ----------
        q_s : one of torch.distributions.Distribution
            Variational distribution over state.
        p_s : one of torch.distributions.Distribution
            Pior distribution over state.
        o_t : torch.Tensor
            Shape (batch_size, o_dim)
        p_o : one of torch.distributions.Distribution
            Distribution over observation.
        a_t : torch.Tensor
            Shape (batch_size, a_dim)
        p_a : one of torch.distributions.Distribution
            Distribution over action.
        """
        # KL divergence between posterior and prior of states
        kl = kl_divergence(q_s, p_s).mean(dim=0).sum()
        # negative log likelihood of observation
        nll_o = - p_o.log_prob(o_t).mean(dim=0).sum()
        # negative log likelihood of action
        nll_a = - p_a.log_prob(a_t).mean(dim=0).sum()
        return kl + nll_o + nll_a

    def take_action(self, obs):
        o_t = torch.tensor([obs], dtype=torch.float)
        s_tm1 = self.s_t
        a_t = self.a_t
        s_t, a_tp1, q_s, p_s, p_o, p_a = self.apl(s_tm1, o_t, a_t)

        self.s_t = s_t.mean(dim=0)
        self.a_t = a_tp1.mean(dim=0)
        self.q_s = q_s
        self.p_s = p_s
        self.p_o = p_o
        self.p_a = p_a
        fe_t = self._calc_FE(q_s, p_s, o_t, p_o, a_t, p_a)
        action_probs = F.softmax(self.a_t, dim=1)
        action = Categorical(action_probs).sample().item()
        return action, fe_t

    def train(self, n_episodes, n_steps):
        self.apl.training = True
        for i_episode in range(n_episodes):
            obs = self.env.reset()
            fe = 0
            self.s_t = self.s_0
            self.a_t = self.a_0
            t = 0

            for t in range(n_steps):
                #self.env.render()
                action, fe_inc = self.take_action(obs)
                obs, reward, done, info = self.env.step(action)
                fe = torch.add(fe, fe_inc)

                if (t + 1) % 100 == 0:
                    print("timestep {}, free energy {:.3f}".format(t, fe.item()))
                    fe.backward(retain_graph=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                elif done:
                    fe.backward(retain_graph=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    break

            print("episode: {}, total timesteps: {}, avg free energy {:.3f}".format(i_episode+1, t+1, fe.item() / t+1))
            print("=================================================")

        self.env.close()
        self.apl.training = False
