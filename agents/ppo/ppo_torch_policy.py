import logging

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from gym.spaces import Box, Discrete
from agents.utils import tanh_to_action, normalize_obs
from policy.actor_critic import MLPGaussianActor, MLPCategoricalActor, MLPCritic
from policy.torch_policy import TorchPolicy
from ray.rllib import Policy, SampleBatch
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.utils import override
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from torch.distributions import OneHotCategorical
from torch.optim import Adam

from agents.diayn import MLP
from policy.postprocessing import Postprocessing, compute_advantages


ACTIVATIONS = "activations"
SKILLS = "skills"
SKILL_LOGQ = "skill_logq"


logger = logging.getLogger(__name__)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MLPDiscriminator(nn.Module):

    """Estimate log p(z | s)."""

    def __init__(self, in_dim, out_dim, hidden_sizes, **kwargs):
        super(MLPDiscriminator, self).__init__()
        self.network = MLP(in_dim, out_dim, hidden_sizes, **kwargs)
        self.out_dim = out_dim

    def forward(self, s):
        if self.out_dim == 1:
            return torch.log(torch.sigmoid(self.network(s)))
        return nn.functional.log_softmax(self.network(s), dim=1)


class SkilledA2C(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh,
                 skills=None):
        super().__init__()

        self.skills = skills
        num_skills = skills.event_shape[0] if skills.event_shape else 0
        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim + num_skills, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim + num_skills, action_space.n, hidden_sizes, activation)

        # build value function
        self.vf = MLPCritic(obs_dim + num_skills, hidden_sizes, activation)

        # set up skill discriminator
        if self.skills is not None:
            self.discriminator = MLPDiscriminator(obs_dim, num_skills, hidden_sizes)

    def step(self, obs, z=None):
        if self.skills is not None:
            assert isinstance(z, torch.Tensor), "self.skills={} but skill value is {}".format(self.skills, z)
            with torch.no_grad():
                oz = torch.cat([obs, z], dim=-1)

                # Compute policy
                pi = self.pi._distribution(oz)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)

                # Compute value
                v = self.vf(oz)

                # log P(z | s)
                logq_z = self.discriminator(obs).numpy()
            return a.numpy(), v.numpy(), logp_a.numpy(), logq_z
        else:
            with torch.no_grad():
                # Compute policy
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)

                # Compute value
                v = self.vf(obs)
            return a.numpy(), v.numpy(), logp_a.numpy(), None

    def act(self, obs, z=None):
        return self.step(obs, z)[0]


class PPOTorchPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.device = torch.device('cpu')

        # Get hyperparameters
        self.alpha = config['alpha']
        self.clip_ratio = config['clip_ratio']
        self.gamma = config['gamma']
        self.lam = config['lambda']
        self.lr_pi = config['lr_pi']
        self.lr_vf = config['lr_vf']
        self.model_hidden_sizes = config['model_hidden_sizes']
        self.num_skills = config['num_skills']
        self.skill_input = config['skill_input']
        self.target_kl = config['target_kl']
        self.use_diayn = config['use_diayn']
        self.use_env_rewards = config['use_env_rewards']
        self.use_gae = config['use_gae']

        # Initialize actor-critic model
        self.skills = OneHotCategorical(torch.ones((1, self.num_skills)))
        if self.skill_input is not None:
            skill_vec = [0.] * (self.num_skills - 1)
            skill_vec.insert(self.skill_input, 1.)
            self.z = torch.as_tensor([skill_vec], dtype=torch.float32)
        else:
            self.z = None
        self.model = SkilledA2C(observation_space,
                                action_space,
                                hidden_sizes=self.model_hidden_sizes,
                                skills=self.skills).to(self.device)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.model.pi.parameters(), self.lr_pi)
        self.vf_optimizer = Adam(self.model.vf.parameters(), self.lr_vf)
        self.disc_optimizer = Adam(self.model.discriminator.parameters(), self.lr_vf)

    def compute_loss_d(self, batch):
        obs, z = batch[SampleBatch.CUR_OBS], batch[SKILLS]
        logq_z = self.model.discriminator(obs)
        return nn.functional.nll_loss(logq_z, z.argmax(dim=-1))

    def compute_loss_pi(self, batch):
        obs, act, z = batch[SampleBatch.CUR_OBS], batch[ACTIVATIONS], batch[SKILLS]
        adv, logp_old = batch[Postprocessing.ADVANTAGES], batch[SampleBatch.ACTION_LOGP]
        clip_ratio = self.clip_ratio

        # Policy loss
        oz = torch.cat([obs, z], dim=-1)
        pi, logp = self.model.pi(oz, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clip_frac)

        return loss_pi, pi_info

    def compute_loss_v(self, batch):
        obs, z = batch[SampleBatch.NEXT_OBS], batch[SKILLS]
        v_pred_old, v_targ = batch[SampleBatch.VF_PREDS], batch[Postprocessing.VALUE_TARGETS]

        oz = torch.cat([obs, z], dim=-1)
        v_pred = self.model.vf(oz)
        v_pred_clipped = v_pred_old + torch.clamp(v_pred - v_pred_old, -self.clip_ratio, self.clip_ratio)

        loss_clipped = (v_pred_clipped - v_targ).pow(2)
        loss_unclipped = (v_pred - v_targ).pow(2)

        return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

    def _convert_activation_to_action(self, activation):
        min_ = self.action_space.low
        max_ = self.action_space.high
        return tanh_to_action(activation, min_, max_)

    def _normalize_obs(self, obs):
        min_ = self.observation_space.low
        max_ = self.observation_space.high
        return normalize_obs(obs, min_, max_)

    @override(Policy)
    def compute_actions(self,
                        obs,
                        **kwargs):
        # Sample a skill at the start of each episode
        if self.z is None:
            self.z = self.skills.sample()

        o = self._normalize_obs(obs)
        a, v, logp_a, logq_z = self.model.step(torch.as_tensor(o, dtype=torch.float32), self.z)

        actions = self._convert_activation_to_action(a)
        extras = {
            ACTIVATIONS: a,
            SampleBatch.VF_PREDS: v,
            SampleBatch.ACTION_LOGP: logp_a,
            SKILLS: self.z.numpy(),
            SKILL_LOGQ: logq_z
        }
        return actions, [], extras

    @override(Policy)
    def postprocess_trajectory(self,
                               batch,
                               other_agent_batches=None,
                               episode=None):
        """Adds the policy logits, VF preds, and advantages to the trajectory."""

        completed = batch["dones"][-1]
        if completed:
            # Force end of episode reward
            last_r = 0.0

            # Reset skill at the end of each episode
            self.z = None
        else:
            next_state = []
            for i in range(self.num_state_tensors()):
                next_state.append([batch["state_out_{}".format(i)][-1]])
            obs = [batch[SampleBatch.NEXT_OBS][-1]]
            o = self._normalize_obs(obs)
            _, last_r, _, _ = self.model.step(torch.as_tensor(o, dtype=torch.float32), self.z)
            last_r = last_r.item()

        # Compute DIAYN rewards
        if self.use_diayn:
            z = torch.as_tensor(batch[SKILLS], dtype=torch.float32)
            logp_z = self.skills.log_prob(z).numpy()
            logq_z = batch[SKILL_LOGQ][:, z.argmax(dim=-1)[0].item()]
            entropy_reg = self.alpha * batch[SampleBatch.ACTION_LOGP]
            diayn_rewards = logq_z - logp_z - entropy_reg

            if self.use_env_rewards:
                batch[SampleBatch.REWARDS] += diayn_rewards
            else:
                batch[SampleBatch.REWARDS] = diayn_rewards

        batch = compute_advantages(
            batch,
            last_r,
            gamma=self.gamma,
            lambda_=self.lam,
            use_gae=self.use_gae)
        return batch

    @override(Policy)
    def learn_on_batch(self, postprocessed_batch):
        postprocessed_batch[SampleBatch.CUR_OBS] = self._normalize_obs(postprocessed_batch[SampleBatch.CUR_OBS])
        train_batch = self._lazy_tensor_dict(postprocessed_batch)

        # Train policy with multiple steps of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(train_batch)
        # if pi_info['kl'] > 1.5 * self.target_kl:
        #     logger.info('Early stopping at step %d due to reaching max kl.' % i)
        #     return
        loss_pi.backward()
        self.pi_optimizer.step()

        # Value function learning
        self.vf_optimizer.zero_grad()
        loss_v = self.compute_loss_v(train_batch)
        loss_v.backward()
        self.vf_optimizer.step()

        # Discriminator learning
        self.disc_optimizer.zero_grad()
        loss_d = self.compute_loss_d(train_batch)
        loss_d.backward()
        self.disc_optimizer.step()

        grad_info = dict(
            pi_loss=loss_pi.item(),
            vf_loss=loss_v.item(),
            d_loss=loss_d.item(),
            **pi_info
        )
        return {LEARNER_STATS_KEY: grad_info}
