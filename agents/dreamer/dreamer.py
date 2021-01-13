import logging

import random
import numpy as np

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, LEARNER_INFO, _get_shared_metrics
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import StoreToReplayBuffer, Replay
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.utils.typing import SampleBatchType

from agents.dreamer.dreamer_model import DreamerModel
from agents.dreamer.dreamer_torch import DreamerTorchPolicy

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    "framework": "torch",
    # PlaNET Model LR
    "td_model_lr": 6e-4,
    # Actor LR
    "actor_lr": 8e-5,
    # Critic LR
    "critic_lr": 8e-5,
    # Grad Clipping
    "grad_clip": 100.0,
    # Discount
    "discount": 0.99,
    # Lambda
    "lambda": 0.95,
    # Training iterations per rollout
    "timesteps_per_iteration": 100,
    # Number of episodes in buffer
    "buffer_size": 10000,
    # Number of rollouts to replay
    "batch_size": 50,
    # Length of each rollout
    "rollout_fragment_length": 50,
    # Imagination Horizon for Training Actor and Critic
    "imagine_horizon": 15,
    # Free Nats
    "free_nats": 3.0,
    # KL Coeff for the Model Loss
    "kl_coeff": 1.0,
    # Distributed Dreamer not implemented yet
    "num_workers": 0,
    # Prefill Timesteps
    "learning_starts": 5000,
    # This should be kept at 1 to preserve sample efficiency
    "num_envs_per_worker": 1,
    # Exploration Gaussian
    "explore_noise": 1.0,
    # Batch mode
    "batch_mode": "complete_episodes",
    # Custom Model
    "dreamer_model": {
        "custom_model": DreamerModel,
        # RSSM/PlaNET parameters
        "deter_size": 200,
        "stoch_size": 30,
        # CNN Decoder Encoder
        "depth_size": 32,
        # General Network Parameters
        "hidden_size": 300,
        # Action STD
        "action_init_std": 5.0,
    },
    "reward_scale": 1,

    # === Environment Settings ===
    # Discount factor of the MDP.
    "gamma": 0.99,
    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": None,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": False,
    # Don't set 'done' at the end of the episode. Note that you still need to
    # set this if soft_horizon=True, unless your env is actually running
    # forever without returning done=True.
    "no_done_at_end": False,
    # Arguments to pass to the env creator.
    "env_config": {},
    # Environment name can also be passed via config.
    "env": None,
    # Unsquash actions to the upper and lower bounds of env's action space
    "normalize_actions": False,
    # Whether to clip rewards during Policy's postprocessing.
    # None (default): Clip for Atari only (r=sign(r)).
    # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
    # False: Never clip.
    # [float value]: Clip at -value and + value.
    # Tuple[value1, value2]: Clip at value1 and value2.
    "clip_rewards": None,
    # Whether to clip actions to the action space's low/high range spec.
    "clip_actions": False,
    # Whether to use "rllib" or "deepmind" preprocessors by default
    "preprocessor_pref": "deepmind",
    # The default learning rate.
    "lr": 0.0001,
})
# __sphinx_doc_end__
# yapf: enable


class EpisodicBuffer(object):
    def __init__(self,
                 buffer_size: int = 1000,
                 rollout_length: int = 50,
                 batch_size: int = 50,
                 learning_starts: int = 1000):
        """Data structure that stores episodes and samples chunks
        of size length from episodes

        Args:
            max_length: Maximum episodes it can store
            length: Episode chunking lengh in sample()
        """

        # Stores all episodes into a list: List[SampleBatchType]
        self.episodes = []
        self.buffer_size = buffer_size
        self.timesteps = 0
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.learning_starts = learning_starts

    def add_batch(self, batch: SampleBatchType):
        """Splits a SampleBatch into episodes and adds episodes
        to the episode buffer

        Args:
            batch: SampleBatch to be added
        """

        self.timesteps += batch.count
        episodes = batch.split_by_episode()

        for i, e in enumerate(episodes):
            episodes[i] = self.preprocess_episode(e)
        self.episodes.extend(episodes)

        if len(self.episodes) > self.buffer_size:
            delta = len(self.episodes) - self.buffer_size
            # Drop oldest episodes
            self.episodes = self.episodes[delta:]

    def preprocess_episode(self, episode: SampleBatchType):
        """Batch format should be in the form of (s_t, a_(t-1), r_(t-1))
        When t=0, the resetted obs is paired with action and reward of 0.

        Args:
            episode: SampleBatch representing an episode
        """
        obs = episode["obs"]
        new_obs = episode["new_obs"]
        action = episode["actions"]
        reward = episode["rewards"]

        act_shape = action.shape
        act_reset = np.array([0.0] * act_shape[-1])[None]
        rew_reset = np.array(0.0)[None]
        obs_end = np.array(new_obs[act_shape[0] - 1])[None]

        batch_obs = np.concatenate([obs, obs_end], axis=0)
        batch_action = np.concatenate([act_reset, action], axis=0)
        batch_rew = np.concatenate([rew_reset, reward], axis=0)

        new_batch = {
            "obs": batch_obs,
            "rewards": batch_rew,
            "actions": batch_action
        }
        return SampleBatch(new_batch)

    def replay(self):
        """Samples [batch_size, length] from the list of episodes

        Args:
            batch_size: batch_size to be sampled
        """

        if self.timesteps < self.learning_starts:
            return None

        episodes_buffer = []
        while len(episodes_buffer) < self.batch_size:
            rand_index = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[rand_index]
            if episode.count < self.rollout_length:
                continue
            available = episode.count - self.rollout_length
            index = int(random.randint(0, available))
            episodes_buffer.append(episode.slice(index, index + self.rollout_length))

        batch = {}
        for k in episodes_buffer[0].keys():
            batch[k] = np.stack([e[k] for e in episodes_buffer], axis=0)

        return SampleBatch(batch)


def total_sampled_timesteps(worker):
    return worker.policy_map[DEFAULT_POLICY_ID].global_timestep


def execution_plan(workers, config):
    # Special Replay Buffer for Dreamer agent
    episode_buffer = EpisodicBuffer(
        buffer_size=config["buffer_size"],
        rollout_length=config["rollout_fragment_length"],
        batch_size=config["batch_size"],
        learning_starts=config["learning_starts"])

    local_worker = workers.local_worker()

    # Prefill episode buffer with initial exploration (uniform sampling)
    # while total_sampled_timesteps(local_worker) < config["learning_starts"]:
    #     samples = local_worker.sample()
    #     episode_buffer.add_batch(samples)

    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    store_op = rollouts.for_each(StoreToReplayBuffer(local_buffer=episode_buffer))

    replay_op = Replay(local_buffer=episode_buffer).for_each(TrainOneStep(workers))

    train_op = Concurrently(
        [store_op, replay_op],
        mode="round_robin",
        output_indexes=[1],
        round_robin_weights=[1, config["timesteps_per_iteration"]])
    return StandardMetricsReporting(train_op, workers, config)


def get_policy_class(config):
    return DreamerTorchPolicy


def validate_config(config):
    config["action_repeat"] = 1
    if config["framework"] != "torch":
        raise ValueError("Dreamer not supported in Tensorflow yet!")
    if config["batch_mode"] != "complete_episodes":
        raise ValueError("truncate_episodes not supported")
    # if config["num_workers"] != 0:
    #     raise ValueError("Distributed Dreamer not supported yet!")
    if config["clip_actions"]:
        raise ValueError("Clipping is done inherently via policy tanh!")
    # if config["action_repeat"] > 1:
    #     config["horizon"] = config["horizon"] / config["action_repeat"]


DreamerTrainer = build_trainer(
    name="Dreamer",
    default_config=DEFAULT_CONFIG,
    default_policy=DreamerTorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=execution_plan,
    validate_config=validate_config)
