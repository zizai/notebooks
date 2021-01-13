import logging
import numpy as np
import torch
from chaosbreaker.spaces import FloatBox
from gym.spaces import Discrete
from agents.sync_batch_replay_optimizer import SyncBatchReplayOptimizer
from agents.utils import tanh_to_action, normalize_obs
from models.rssm import RSSMState, get_feat, get_dist
from policy.dreamer import Dreamer
from policy.postprocessing import compute_advantages, Postprocessing
from policy.actor_critic import SAC
from policy.torch_policy import TorchPolicy
from utils.buffer import buffer_method
from utils.module import FreezeParameters
from ray.rllib import SampleBatch
from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

ACTIVATIONS = "activations"
REWARD_PREDS = "reward_preds"
DEFAULT_CONFIG = with_common_config({
    # Model-based dynamics
    "use_dynamics": True,
    # Lagragian dynamics
    "use_lagrangian": True,
    # Planning horizon for model-based control
    "plan_horizon": 20,
    # GAE estimator
    "use_gae": True,
    # Hidden dim for models
    "hidden_dim": 200,
    # Reward discount factor
    "gamma": 0.99,
    # The GAE(lambda) parameter.
    "lambda": 0.97,
    # Learning rate
    "lr": 3e-4,
    # Learning rate schedule
    "lr_schedule": None,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 100,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.5,
    # Entropy coefficient
    "entropy_coeff": 0.01,
    # Minimum KL divergence
    "free_nats": 3,
    # KL scaling factor,
    "kl_scale": 0.1,
    # Number of SGD iterations per train step
    "num_sgd_iter": 1,
    # SGD minibatch size, 0 means no minibatch
    "sgd_minibatch_size": 0,
    # Rollout sequence length
    "rollout_fragment_length": 50,
    # Replay buffer size
    "buffer_size": 1e6,
    # Burn-in for replay
    "learning_starts": 1e4,
    # Batch size per train step
    "train_batch_size": 2500,
    "train_every": 1000,
    # Turn on explore param in compute_actions
    "explore": True,
    "max_episode_len": 1000,
    "policy": "default",
    "num_workers": 2,
    "seed": 123,
    "monitor": False
})


def compute_return(reward: torch.Tensor,
                   value: torch.Tensor,
                   discount: torch.Tensor,
                   bootstrap: torch.Tensor,
                   lambda_: float):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


class A2CTorchPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.use_dynamics = config["use_dynamics"]
        self.use_gae = config["use_gae"]
        self.use_lagrangian = config["use_lagrangian"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters
        self.gamma = config["gamma"]
        self.lambda_ = config["lambda"]
        self.grad_clip = config["grad_clip"]
        self.lr = config["lr"]
        self.vf_loss_coeff = config["vf_loss_coeff"]
        self.entropy_coeff = config["entropy_coeff"]
        self.free_nats = config["free_nats"]
        self.kl_scale = config["kl_scale"]
        self.plan_horizon = config["plan_horizon"]
        self.rollout_fragment_length = config["rollout_fragment_length"]

        self.reward_scale = 100
        self.train_noise = 0.4
        self.eval_noise = 0.
        self.expl_min = 0.
        self.expl_decay = 7000
        self.local_timestep = 0

        self.discrete = isinstance(action_space, Discrete)
        self.expl_type = "additive_gaussian" if not self.discrete else "epsilon_greedy"
        action_dim = action_space.shape[0] if not self.discrete else action_space.n
        hidden_dim = config["hidden_dim"]

        if self.use_dynamics:
            self.model = Dreamer(observation_space, action_dim, hidden_dim=hidden_dim, discrete=self.discrete)
            self.dynamics_model = nn.ModuleList((
                self.model.encoder,
                self.model.decoder,
                self.model.transition,
                self.model.representation,
                self.model.reward
            ))
            self.dynamics_optimizer = Adam(self.dynamics_model.parameters(), self.lr)
        else:
            if isinstance(observation_space, FloatBox):
                obs_dim = observation_space.shape[0]
            else:
                raise ValueError(observation_space)
            self.model = SAC(obs_dim, action_dim, hidden_sizes=(hidden_dim, hidden_dim))
            self.dynamics_model = None
            self.dynamics_optimizer = None

        self.policy_model = nn.ModuleList((
            self.model.pi,
            self.model.vf
        ))
        self.policy_optimizer = Adam(self.policy_model.parameters(), self.lr)

    def _convert_activation_to_action(self, activation):
        min_ = self.action_space.low
        max_ = self.action_space.high
        return tanh_to_action(activation, min_, max_)

    def _normalize_obs(self, obs):
        min_ = self.observation_space.low
        max_ = self.observation_space.high
        return normalize_obs(obs, min_, max_)

    def _exploration(self, action):
        """
        :param action: action to take, shape (1,) (if categorical), or (action dim,) (if continuous)
        :return: action of the same shape passed in, augmented with some noise
        """
        expl_amount = self.train_noise
        if self.expl_decay:  # Linear decay
            expl_amount = expl_amount - self.local_timestep / self.expl_decay
        expl_amount = max(self.expl_min, expl_amount, 0)

        if self.expl_type == "additive_gaussian":  # For continuous actions
            noise = torch.randn(*action.shape, device=action.device) * expl_amount
            return torch.clamp(action + noise, -1, 1)
        if self.expl_type == "completely_random":  # For continuous actions
            if expl_amount == 0:
                return action
            else:
                return torch.rand(*action.shape, device=action.device) * 2 - 1  # scale to [-1, 1]
        if self.expl_type == "epsilon_greedy":  # For discrete actions
            action_dim = self.env_model_kwargs["action_shape"][0]
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, action_dim, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[..., index] = 1
            return action
        raise NotImplementedError(self.expl_type)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_initial_state(self):
        logger.info("get initial state")
        if self.use_dynamics:
            rssm_state = self.model.representation.initial_state(1, device=self.device, dtype=torch.float)
            return [i.squeeze(0).cpu().numpy() for i in rssm_state]
        else:
            return []

    @torch.no_grad()
    def compute_actions(self,
                        obs,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        self.model.cpu()
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        o = self._normalize_obs(obs)
        o = torch.tensor(o, dtype=torch.float)
        if self.use_dynamics:
            prev_action = torch.tensor(prev_action_batch, dtype=torch.float)
            state_batches = [torch.tensor(d, dtype=torch.float) for d in state_batches]
            prev_state = RSSMState(*state_batches)
            a, pi, v, r_pred, state = self.model(o, prev_action=prev_action, prev_state=prev_state)
            if explore:
                a = self._exploration(a)
                self.local_timestep += 1
            action = self._convert_activation_to_action(a.numpy())
            state = [i.numpy() for i in state]
            extras = {
                ACTIVATIONS: a.numpy(),
            }
        else:
            a, v, logp_a = self.model(o)
            action = self._convert_activation_to_action(a.numpy())
            state = []
            extras = {
                ACTIVATIONS: a.numpy(),
                SampleBatch.VF_PREDS: v.numpy(),
                SampleBatch.ACTION_LOGP: logp_a.numpy(),
            }

        return action, state, extras

    @torch.no_grad()
    def postprocess_trajectory(self,
                               batch,
                               other_agent_batches=None,
                               episode=None):
        if self.use_dynamics:
            return batch

        batch[SampleBatch.REWARDS] *= self.reward_scale

        completed = batch[SampleBatch.DONES][-1]
        if completed:
            last_r = 0.0
        else:
            obs = [batch[SampleBatch.NEXT_OBS][-1]]
            o = self._normalize_obs(obs)
            o = torch.tensor(o, dtype=torch.float)
            _, v, _ = self.model(o)
            last_r = v.item()

        return compute_advantages(batch, last_r, self.gamma, self.lambda_, self.use_gae)

    def _compute_loss_pi(self, batch):
        obs, act = batch[SampleBatch.CUR_OBS], batch[ACTIVATIONS]
        adv, logp_old = batch[Postprocessing.ADVANTAGES], batch[SampleBatch.ACTION_LOGP]

        # Loss pi
        pi, logp = self.model.pi(obs, act)
        loss_pi = - adv.dot(logp.reshape(-1))

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        return loss_pi, pi_info

    def _compute_loss_v(self, batch):
        obs = batch[SampleBatch.NEXT_OBS]
        v_pred_old, v_targ = batch[SampleBatch.VF_PREDS], batch[Postprocessing.VALUE_TARGETS]

        v_pred = self.model.vf(obs)
        loss_v = nn.functional.mse_loss(v_pred.reshape(-1), v_targ)
        return loss_v

    def learn_on_batch(self, postprocessed_batch):
        postprocessed_batch[SampleBatch.CUR_OBS] = self._normalize_obs(postprocessed_batch[SampleBatch.CUR_OBS])
        train_batch = self._lazy_tensor_dict(postprocessed_batch)
        logger.info("train step!")
        self.model.to(self.device)

        if self.use_dynamics:
            o, prev_a, r = train_batch[SampleBatch.CUR_OBS], train_batch[ACTIVATIONS], train_batch[SampleBatch.REWARDS]
            batch_size = o.shape[0]
            T = self.rollout_fragment_length
            B = int(batch_size / T)

            # Extract tensors from the Samples object
            # They all have the batch_t dimension first, but we'll put the batch_b dimension first.
            # Also, we convert all tensors to floats so they can be fed into our models.

            o = o.reshape(T, B, -1)
            prev_a = prev_a.reshape(T, B, -1)
            r = r.reshape(T, B, -1)

            o = o[:-1]
            prev_a = prev_a[1:]
            r = r[1:]
            T = o.shape[0]
            batch_size = T * B

            embed = self.model.encoder(o)

            prev_state = self.model.representation.initial_state(B, device=prev_a.device, dtype=prev_a.dtype)
            # Rollout model by taking the same series of actions as the real model
            post, prior = self.model.rollout.rollout_representation(T, embed, prev_a, prev_state)
            # Flatten our data (so first dimension is batch_t * batch_b = batch_size)
            # since we're going to do a new rollout starting from each state visited in each batch.

            # Compute losses for each component of the model

            # Model Loss
            feat = get_feat(post)
            o_pred = self.model.decoder(feat)
            o_pred = torch.distributions.Normal(o_pred, 1)
            r_pred = self.model.reward(feat)
            r_loss = -torch.mean(r_pred.log_prob(r))
            o_loss = -torch.mean(o_pred.log_prob(o))
            prior_dist = get_dist(prior)
            post_dist = get_dist(post)
            div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            div = torch.max(div, div.new_full(div.size(), self.free_nats))
            dynamics_loss = self.kl_scale * div + r_loss + o_loss

            # ------------------------------------------  Gradient Barrier  ------------------------------------------------
            # Don't let gradients pass through to prevent overwriting gradients.
            # Actor Loss

            # remove gradients from previously calculated tensors
            with torch.no_grad():
                flat_post = buffer_method(post, "reshape", batch_size, -1)
            # Rollout the policy for self.horizon steps.
            # Variable names with imag_ indicate this data is imagined not real.
            # imag_feat shape is [horizon, batch_t * batch_b, feature_size]
            with FreezeParameters([self.model.encoder, self.model.decoder, self.model.transition, self.model.representation, self.model.reward]):
                imag_dist, _ = self.model.rollout.rollout_policy(self.plan_horizon, self.model.policy, flat_post)

            # Use state features (deterministic and stochastic) to predict the image and reward
            imag_feat = get_feat(imag_dist)  # [horizon, batch_t * batch_b, feature_size]
            # Assumes these are normal distributions.
            # If we want to use other distributions we'll have to fix this.
            # We calculate the target here so no grad necessary

            # freeze model parameters as only action model gradients needed
            with FreezeParameters([self.model.encoder, self.model.decoder, self.model.transition, self.model.representation, self.model.reward, self.model.vf]):
                imag_reward = self.model.reward(imag_feat).mean
                value = self.model.vf(imag_feat).mean

            # Compute the exponential discounted sum of rewards
            discount_arr = self.gamma * torch.ones_like(imag_reward)
            returns = compute_return(imag_reward[:-1], value[:-1], discount_arr[:-1], value[-1], self.lambda_)
            # Make the top row 1 so the cumulative product starts with discount^0
            discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
            discount = torch.cumprod(discount_arr[:-1], 0)
            pi_loss = -torch.mean(discount * returns)

            # ------------------------------------------  Gradient Barrier  ------------------------------------------------
            # Don't let gradients pass through to prevent overwriting gradients.
            # Value Loss

            # remove gradients from previously calculated tensors
            with torch.no_grad():
                value_feat = imag_feat[:-1].detach()
                value_discount = discount.detach()
                value_target = returns.detach()
            value_pred = self.model.vf(value_feat)
            log_prob = value_pred.log_prob(value_target)
            v_loss = -torch.mean(value_discount * log_prob.unsqueeze(2))

            policy_loss = sum([
                pi_loss,
                self.vf_loss_coeff * v_loss
            ])
            self.dynamics_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()

            dynamics_loss.backward()
            policy_loss.backward()

            grad_norm_dynamics = torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.grad_clip)
            grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)

            self.dynamics_optimizer.step()
            self.policy_optimizer.step()
            # ------------------------------------------  Gradient Barrier  ------------------------------------------------
            # loss info
            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                post_ent = torch.mean(post_dist.entropy())

                grad_info = dict(
                    dynamics_loss=dynamics_loss.item(),
                    o_loss=o_loss.item(),
                    r_loss=r_loss.item(),
                    kl=div.item(),
                    prior_ent=prior_ent.item(),
                    post_ent=post_ent.item(),
                    policy_loss=policy_loss.item(),
                    pi_loss=pi_loss.item(),
                    vf_loss=v_loss.item(),
                )

                if self.config["monitor"]:
                    writer = SummaryWriter(log_dir=self.config["log_dir"])
                    openl = torch.rand((4, 20, 3, 100, 100), dtype=torch.float) # N,T,C,H,W
                    video = torch.clamp(openl, 0., 1.)
                    writer.add_video(tag="videos", vid_tensor=video, global_step=self.global_timestep, fps=20)
                    logger.info("video logged to: {}".format(self.config["log_dir"]))

            return {LEARNER_STATS_KEY: grad_info}
        else:
            self.policy_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_loss_pi(train_batch)
            loss_v = self._compute_loss_v(train_batch)
            policy_loss = sum([
                loss_pi,
                self.vf_loss_coeff * loss_v,
                - self.entropy_coeff * pi_info["ent"],
                ])
            policy_loss.backward()
            grad_norm_policy = nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
            self.policy_optimizer.step()

            grad_info = dict(
                pi_loss=loss_pi.item(),
                vf_loss=loss_v.item(),
                grad_norm=grad_norm_policy.item() if isinstance(grad_norm_policy, torch.Tensor) else grad_norm_policy,
                **pi_info
            )
            return {LEARNER_STATS_KEY: grad_info}

    @property
    def num_params(self):
        return sum([p.numel() for p in self.model.parameters()])


def before_init(trainer):
    trainer.config["log_dir"] = trainer._logdir


def make_policy_optimizer(workers, config):
    optimizer = SyncBatchReplayOptimizer(workers,
                                         learning_starts=config["learning_starts"],
                                         buffer_size=config["buffer_size"],
                                         train_batch_size=config["train_batch_size"],
                                         num_sgd_iter=config["num_sgd_iter"],
                                         train_every=config["train_every"])
    return optimizer


A2CTrainer = build_trainer(
    name="A2C",
    default_config=DEFAULT_CONFIG,
    default_policy=A2CTorchPolicy,
    before_init=before_init,
    make_policy_optimizer=make_policy_optimizer)
