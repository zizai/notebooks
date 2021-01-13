import logging

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from agents.ppo.ppo_torch_policy import ACTIVATIONS, PPOTorchPolicy
from policy.neuroblast_policy import NeuroblastA2C
from ray.rllib import Policy, SampleBatch
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.utils import override
from torch.optim import Adam

from policy.postprocessing import Postprocessing, compute_advantages


logger = logging.getLogger(__name__)


class PPONeuroblastPolicy(PPOTorchPolicy):
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
        self.use_env_rewards = config['use_env_rewards']
        self.use_gae = config['use_gae']

        # Initialize actor-critic model
        self.model = NeuroblastA2C(observation_space,
                                   action_space).to(self.device)
        self.genome = self.model.genome

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.model.pi.parameters(), self.lr_pi)
        self.vf_optimizer = Adam(self.model.vf.parameters(), self.lr_vf)

    def compute_loss_pi(self, batch):
        obs, act = batch[SampleBatch.CUR_OBS], batch[ACTIVATIONS]
        adv, logp_old = batch[Postprocessing.ADVANTAGES], batch[SampleBatch.ACTION_LOGP]
        clip_ratio = self.clip_ratio

        # Policy loss
        pi, logp = self.model.pi(obs, act)
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
        obs = batch[SampleBatch.NEXT_OBS]
        v_pred_old, v_targ = batch[SampleBatch.VF_PREDS], batch[Postprocessing.VALUE_TARGETS]

        v_pred = self.model.vf(obs)
        v_pred_clipped = v_pred_old + torch.clamp(v_pred - v_pred_old, -self.clip_ratio, self.clip_ratio)

        loss_clipped = (v_pred_clipped - v_targ).pow(2)
        loss_unclipped = (v_pred - v_targ).pow(2)

        return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

    def compute_actions(self,
                        obs,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        **kwargs):
        o = self._normalize_obs(obs)
        a, v, logp_a = self.model.step(torch.as_tensor(o, dtype=torch.float32))

        actions = self._convert_activation_to_action(a.numpy())
        extras = {
            ACTIVATIONS: a.numpy(),
            SampleBatch.VF_PREDS: v.numpy(),
            SampleBatch.ACTION_LOGP: logp_a.numpy(),
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
        else:
            next_state = []
            for i in range(self.num_state_tensors()):
                next_state.append([batch["state_out_{}".format(i)][-1]])
            obs = [batch[SampleBatch.NEXT_OBS][-1]]
            o = self._normalize_obs(obs)
            _, last_r, _ = self.model.step(torch.as_tensor(o, dtype=torch.float32))
            last_r = last_r.item()

        batch = compute_advantages(
            batch,
            last_r,
            gamma=self.gamma,
            lambda_=self.lam,
            use_gae=self.use_gae)
        return batch

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

        # regenerate phenotype with new decoder
        self.model.update()

    def learn_on_batch(self, postprocessed_batch):
        postprocessed_batch[SampleBatch.CUR_OBS] = self._normalize_obs(postprocessed_batch[SampleBatch.CUR_OBS])
        train_batch = self._lazy_tensor_dict(postprocessed_batch)

        # Train policy with multiple steps of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(train_batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Value function learning
        self.vf_optimizer.zero_grad()
        loss_v = self.compute_loss_v(train_batch)
        loss_v.backward()
        self.vf_optimizer.step()

        # regenerate phenotype with new decoder
        self.model.update()

        grad_info = dict(
            pi_loss=loss_pi.item(),
            vf_loss=loss_v.item(),
            **pi_info
        )
        return {LEARNER_STATS_KEY: grad_info}
