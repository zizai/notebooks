from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import logging
import numpy as np
import time

import ray
from ray.rllib.agents import Trainer, with_common_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.memory import ray_get_and_free

from agents import es_optimizers
from agents.utils import SharedNoiseTable

logger = logging.getLogger(__name__)

Result = namedtuple("Result", [
    "noise_indices", "noisy_returns", "sign_noisy_returns", "noisy_lengths",
    "eval_returns", "eval_lengths"
])

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    "policy": None,
    "l2_coeff": 0.005,
    "noise_stdev": 0.02,
    "episodes_per_batch": 1000,
    "train_batch_size": 10000,
    "eval_prob": 0.003,
    "return_proc_mode": "centered_rank",
    "num_workers": 10,
    "stepsize": 0.01,
    "observation_filter": "MeanStdFilter",
    "noise_size": 250000000,
    "report_length": 10,
})
# __sphinx_doc_end__
# yapf: enable


@ray.remote
def create_shared_noise(count):
    """Create a large array of noise to be shared by all workers."""
    seed = 123
    noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    return noise


@ray.remote
class ESWorker(object):
    def __init__(self,
                 config,
                 policy_cls,
                 policy_params,
                 env_creator,
                 noise,
                 min_task_runtime=0.2):
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.policy_params = policy_params
        self.noise = SharedNoiseTable(noise)
        self.noise_stdev = config['noise_stdev']
        self.eval_prob = config['eval_prob']

        env_config = config['env_config']
        self.env = env_creator(env_config)
        self.policy = policy_cls(self.env.observation_space, self.env.action_space, self.config)

    def rollout(self, timestep_limit):
        rews = []
        t = 0
        obs = self.env.reset()
        for _ in range(timestep_limit or 999999):
            ac = self.policy.compute_single_action(obs)
            obs, rew, done, _ = self.env.step(ac)
            rews.append(rew)
            t += 1
            if done:
                break
        rews = np.array(rews, dtype=np.float32)
        return rews, t

    def do_rollouts(self, params, timestep_limit=None):
        noise_indices, returns, sign_returns, lengths = [], [], [], []
        eval_returns, eval_lengths = [], []

        # Perform some rollouts with noise.
        task_tstart = time.time()
        while (len(noise_indices) == 0
               or time.time() - task_tstart < self.min_task_runtime):

            if np.random.uniform() < self.eval_prob:
                # Do an evaluation run with no perturbation.
                self.policy.set_weights(params)
                rewards, length = self.rollout(timestep_limit)
                eval_returns.append(rewards.sum())
                eval_lengths.append(length)
            else:
                # Do a regular run with parameter perturbations.
                noise_index = self.noise.sample_index(params.size)

                perturbation = self.noise_stdev * self.noise.get(noise_index, params.size)

                self.policy.set_weights(params + perturbation)
                rewards_pos, lengths_pos = self.rollout(timestep_limit)

                self.policy.set_weights(params - perturbation)
                rewards_neg, lengths_neg = self.rollout(timestep_limit)

                noise_indices.append(noise_index)
                returns.append([rewards_pos.sum(0), rewards_neg.sum(0)])
                sign_returns.append([np.sign(rewards_pos).sum(0), np.sign(rewards_neg).sum(0)])
                lengths.append([lengths_pos, lengths_neg])

        return Result(
            noise_indices=noise_indices,
            noisy_returns=returns,
            sign_noisy_returns=sign_returns,
            noisy_lengths=lengths,
            eval_returns=eval_returns,
            eval_lengths=eval_lengths)


class ESTrainer(Trainer):
    """Large-scale implementation of Evolution Strategies in Ray."""

    _name = "ES"
    _default_config = DEFAULT_CONFIG

    @override(Trainer)
    def _init(self, config, env_creator):
        policy_params = {"action_noise_std": 0.01}
        policy_cls = config['policy']

        # Create the shared noise table.
        logger.info("Creating shared noise table.")
        noise_id = create_shared_noise.remote(config["noise_size"])

        # Create the actors.
        logger.info("Creating actors.")
        env_config = config['env_config']
        self.env = env_creator(env_config)
        self._workers = [
            ESWorker.remote(config, policy_cls, policy_params, env_creator, noise_id)
            for _ in range(config["num_workers"])
        ]

        self.policy = policy_cls(self.env.observation_space, self.env.action_space, config)
        self.report_length = config["report_length"]

        # Create optimizer
        shared_noise = ray.get(noise_id)
        l2_coeff = config['l2_coeff']
        stepsize = config['stepsize']
        self.optimizer = es_optimizers.Adam(self.policy.model, shared_noise, l2_coeff, stepsize)

        self.episodes_so_far = 0
        self.reward_list = []
        self.tstart = time.time()
        self.msg_queue = []
        #self.env.register_agent(self)

    def push_messages(self, msg):
        self.msg_queue.append(msg)

    def pull_messages(self):
        msg = self.msg_queue.copy()
        self.msg_queue.clear()
        return msg

    @override(Trainer)
    def _train(self):
        config = self.config

        theta = self.policy.get_weights()

        # Put the current policy weights in the object store.
        theta_id = ray.put(theta)
        # Use the actors to do rollouts, note that we pass in the ID of the
        # policy weights.
        results, num_episodes, num_timesteps = self._collect_results(
            theta_id, config["episodes_per_batch"], config["train_batch_size"])

        all_noise_indices = []
        all_training_returns = []
        all_training_lengths = []
        all_eval_returns = []
        all_eval_lengths = []

        # Loop over the results.
        for result in results:
            all_eval_returns += result.eval_returns
            all_eval_lengths += result.eval_lengths

            all_noise_indices += result.noise_indices
            all_training_returns += result.noisy_returns
            all_training_lengths += result.noisy_lengths

        assert len(all_eval_returns) == len(all_eval_lengths)
        assert (len(all_noise_indices) == len(all_training_returns) ==
                len(all_training_lengths))

        self.episodes_so_far += num_episodes

        # Assemble the results.
        eval_returns = np.array(all_eval_returns)
        eval_lengths = np.array(all_eval_lengths)
        noise_indices = np.array(all_noise_indices)
        noisy_returns = np.array(all_training_returns)
        noisy_lengths = np.array(all_training_lengths)

        # Compute the new weights theta.
        theta, update_ratio = self.optimizer.update(noisy_returns, noise_indices)

        # Store the rewards
        if len(all_eval_returns) > 0:
            self.reward_list.append(np.mean(eval_returns))

        info = {
            "weights_norm": np.square(theta).sum(),
            "update_ratio": update_ratio,
            "episodes_this_iter": noisy_lengths.size,
            "episodes_so_far": self.episodes_so_far,
        }

        reward_mean = np.mean(self.reward_list[-self.report_length:])
        result = dict(
            episode_reward_mean=reward_mean,
            episode_len_mean=eval_lengths.mean(),
            timesteps_this_iter=noisy_lengths.sum(),
            info=info)

        return result

    @override(Trainer)
    def compute_action(self, observation):
        return self.policy.compute_single_action(observation)

    @override(Trainer)
    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for w in self._workers:
            w.__ray_terminate__.remote()

    def _collect_results(self, theta_id, min_episodes, min_timesteps):
        num_episodes, num_timesteps = 0, 0
        results = []
        while num_episodes < min_episodes or num_timesteps < min_timesteps:
            logger.info(
                "Collected {} episodes {} timesteps so far this iter".format(
                    num_episodes, num_timesteps))
            rollout_ids = [
                worker.do_rollouts.remote(theta_id) for worker in self._workers
            ]
            # Get the results of the rollouts.
            for result in ray_get_and_free(rollout_ids):
                results.append(result)
                # Update the number of episodes and the number of timesteps
                # keeping in mind that result.noisy_lengths is a list of lists,
                # where the inner lists have length 2.
                num_episodes += sum(len(pair) for pair in result.noisy_lengths)
                num_timesteps += sum(sum(pair) for pair in result.noisy_lengths)

        return results, num_episodes, num_timesteps

    def __getstate__(self):
        return {
            "model_parameters": self.policy.model.get_parameters(),
            "episodes_so_far": self.episodes_so_far,
        }

    def __setstate__(self, state):
        self.episodes_so_far = state["episodes_so_far"]
        self.policy.model.set_parameters(state["model_parameters"])
