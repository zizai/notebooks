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
class AESWorker(object):
    def __init__(self,
                 config,
                 policy_cls,
                 policy_params,
                 env,
                 noise,
                 min_task_runtime=0.2):
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.policy_params = policy_params
        self.noise = SharedNoiseTable(noise)
        self.noise_stdev = config['noise_stdev']
        self.eval_prob = config['eval_prob']

        self.env = env
        self.policy = policy_cls(config)

    def rollout(self, timestep_limit):
        rews = []
        t = 0
        observation = self.env.reset()
        for _ in range(timestep_limit or 999999):
            ac, d_loss, g_loss = self.policy.compute_single_action(observation)
            observation, rew, done, _ = self.env.step(ac)
            rews.append([-d_loss, -g_loss, rew])
            t += 1
            if done:
                break
        rews = np.array(rews, dtype=np.float32)
        return rews, t

    def do_rollouts(self, params, timestep_limit=None):
        d_params, g_params = params
        noise_indices, returns, sign_returns, lengths = [], [], [], []
        eval_returns, eval_lengths = [], []

        # Perform some rollouts with noise.
        task_tstart = time.time()
        while (len(noise_indices) == 0
               or time.time() - task_tstart < self.min_task_runtime):

            if np.random.uniform() < self.eval_prob:
                # Do an evaluation run with no perturbation.
                self.policy.set_weights(d_params, 'discriminator')
                self.policy.set_weights(g_params, 'generator')
                rewards, length = self.rollout(timestep_limit)
                eval_returns.append(rewards.sum())
                eval_lengths.append(length)
            else:
                # Do a regular run with parameter perturbations.
                d_noise_index = self.noise.sample_index(d_params.size)
                g_noise_index = self.noise.sample_index(g_params.size)

                d_perturbation = self.noise_stdev * self.noise.get(d_noise_index, d_params.size)
                g_perturbation = self.noise_stdev * self.noise.get(g_noise_index, g_params.size)

                self.policy.set_weights(d_params + d_perturbation, 'discriminator')
                self.policy.set_weights(g_params + g_perturbation, 'generator')
                rewards_pos, lengths_pos = self.rollout(timestep_limit)

                self.policy.set_weights(d_params - d_perturbation, 'discriminator')
                self.policy.set_weights(g_params - g_perturbation, 'generator')
                rewards_neg, lengths_neg = self.rollout(timestep_limit)

                noise_indices.append([d_noise_index, g_noise_index])
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


class AESTrainer(Trainer):
    """Adversarial Evolution Strategies"""

    _name = "ES"
    _default_config = DEFAULT_CONFIG

    @override(Trainer)
    def _init(self, config, env_cls):
        policy_params = {"action_noise_std": 0.01}

        policy_cls = config['policy']
        self.policy = policy_cls(config)
        self.report_length = config["report_length"]

        # Create the shared noise table.
        logger.info("Creating shared noise table.")
        noise_id = create_shared_noise.remote(config["noise_size"])
        shared_noise = ray.get(noise_id)
        l2_coeff = config['l2_coeff']
        stepsize = config['stepsize']
        self.d_optimizer = es_optimizers.Adam(self.policy.discriminator, shared_noise, l2_coeff, stepsize)
        self.g_optimizer = es_optimizers.Adam(self.policy.generator, shared_noise, l2_coeff, stepsize)

        # Create the actors.
        logger.info("Creating actors.")
        env_config = config['env_config']
        self.env = env_cls(env_config)

        self._workers = [
            AESWorker.remote(config, policy_cls, policy_params, self.env, noise_id)
            for _ in range(config["num_workers"])
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.tstart = time.time()
        self.msg_queue = []
        self.env.register_agent(self)

    def push_messages(self, msg):
        self.msg_queue.append(msg)

    def pull_messages(self):
        msg = self.msg_queue.copy()
        self.msg_queue.clear()
        return msg

    @override(Trainer)
    def _train(self):
        config = self.config

        theta = [self.policy.get_weights('discriminator'), self.policy.get_weights('generator')]

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

        print(noisy_returns.shape)
        print(noise_indices.shape)
        # Compute the new weights theta.
        d_theta, d_update_ratio = self.d_optimizer.update(noisy_returns[:, :, 0], noise_indices[:, 0])
        g_theta, g_update_ratio = self.g_optimizer.update(noisy_returns[:, :, 1], noise_indices[:, 1])

        # Store the rewards
        if len(all_eval_returns) > 0:
            self.reward_list.append(np.mean(eval_returns))

        info = {
            "weights_norm": [np.square(d_theta).sum(), np.square(g_theta).sum()],
            "update_ratio": [d_update_ratio, g_update_ratio],
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
            "weights": self.policy.get_weights(),
            "episodes_so_far": self.episodes_so_far,
        }

    def __setstate__(self, state):
        self.episodes_so_far = state["episodes_so_far"]
        self.policy.set_weights(state["weights"])
